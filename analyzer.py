from dotenv import load_dotenv
import os
import sys
import pandas as pd
import chardet as cd
import pinecone
import dateparser
import json
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from play_store_review_loader import PlayStoreReviewLoader
import re

load_dotenv()
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

def _load_df(path):
    with open(path, 'rb') as f:
        result = cd.detect(f.read())
        # print(result)
    return pd.read_csv(path, encoding=result['encoding'], on_bad_lines="warn", lineterminator="\n")

def _load_csv(path):
    with open(path, 'rb') as f:
        result = cd.detect(f.read())
    loader = PlayStoreReviewLoader(
        path, 
        encoding=result['encoding']
    )
    return loader.load()

def _upsert_pinecone(docs, embeddings):
    index_name = os.getenv("PINECONE_INDEX")
    print(f'Creating index {index_name}...', index_name) 
    Pinecone.from_documents(docs, embeddings, index_name=index_name)
    pass

def _extract_info_query(llm: BaseLanguageModel, query: str):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        You're a helpful data analyst. Given the following sentence, try to extract day, month and year. 
        Format the date as "DD-MM-YYYY", use 01 as the filler if day is not specified. 
        Apart from getting dates, if rating range is specified, get the rating value too. If not specified, return null instead.
        
        Given sentence: "{query}"
        
        The output should be a JSON object, here is the template:

        {{
            "dates": [..] (string array, nullable),
            "ratings": [..] (as integer array, nullable)
        }}
        """
    )

    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        resp = chain.run(query=query)
        # resp = """
        #     Output: {
        #         "dates": ["01-03-2023"],
        #         "ratings": null
        #     }
        # """

        print(f"_extract_info_query Response={resp}")
        regex_parsed = re.search(r"{[\s\S]*}", resp)
        parsed_resp = regex_parsed.group()
        print(f"parsed_resp={parsed_resp}")

        json_resp = json.loads(parsed_resp)
        print(f"json_resp={json_resp}")

        resp_dict = {
            "dates": None,
            "ratings": None
        }
        print(f"resp_dict={resp_dict}")

        if json_resp["dates"] is not None:
            dts = []
            for dt in json_resp["dates"]:
                epoch_ts = int(dateparser.parse(dt, date_formats=["%d-%m-%Y"]).timestamp() * 1000)
                dts.append(epoch_ts)
            dts.sort()

            resp_dict["dates"] = dts
        else:
            print(json_resp["dates"])

        if json_resp["ratings"] is not None:
            json_resp["ratings"].sort()
            resp_dict["ratings"] = json_resp["ratings"]
        
        print(f"final resp_dict={resp_dict}")

        return resp_dict
    except json.JSONDecodeError:
        print("Failed to parse JSON response.")
        return {}
    except ValueError:
        print("Invalid date format.")
        return {}

def _create_filter(infodict):
    filter_dict = {}

    if infodict["dates"] is not None:
        if len(infodict["dates"]) > 1:
            filter_dict = {
                "$and": [
                    { "last_update_ts": { "$gte": infodict["dates"][0] } }, 
                    { "last_update_ts": { "$lte": infodict["dates"][1] } }
                ]
            }
        else:
            end_dts = infodict["dates"][0] + 2592000000 # +30 days
            filter_dict = {
                "$and": [
                    { "last_update_ts": { "$gte": infodict["dates"][0] } },
                    { "last_update_ts": { "$lte": end_dts } }
                ]
            }

    # Append $and operator if we know that filter_dict has condition already
    if infodict["ratings"] is not None and len(filter_dict) > 0 and "$and" not in filter_dict:
        temp_filter_dict = { "$and": [] }
        for key, value in filter_dict.items():
            temp_filter_dict["$and"].append({ key: value })
        filter_dict = temp_filter_dict

    if infodict["ratings"] is not None:
        if "$and" in filter_dict:
            if len(infodict["ratings"]) > 1:
                filter_dict["$and"].append({"rating": { "$gte": infodict["ratings"][0] }})
                filter_dict["$and"].append({"rating": { "$lte": infodict["ratings"][1] }})
            else:
                filter_dict["$and"].append({"rating": { "$eq": infodict["ratings"][0] }})
        else:            
            if len(infodict["ratings"]) > 1:
                filter_dict = {
                    "$and": [
                        { "rating": { "$gte": infodict["ratings"][0] } }, 
                        { "rating": { "$lte": infodict["ratings"][1] } }
                    ]
                }
            else:
                filter_dict = { "rating": {"$eq": infodict["ratings"][0] } }
    
    if len(filter_dict) == 0:
        print(f"None returned, input={infodict}")
        return None

    print(f"filter_dict={filter_dict}")
    return filter_dict


def _query_llm(index_name, filter_dict, embeddings, llm: BaseLanguageModel, query: str) -> None:
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    if filter_dict is not None:
        print(f"Retrieving with filter, dict={filter_dict}")
        retriever = docsearch.as_retriever(search_kwargs={
            "filter": filter_dict
        })
    else:
        print(f"Retrieving without filter")
        retriever = docsearch.as_retriever()
    
    review_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        verbose=True
    )
    result = review_chain.run(query)
    print(result)

def main():
    index_name = os.getenv("PINECONE_INDEX")
        
    q="""
    You are a data analyst. The data you see are app reviews for the app called 9GAG from Google Play Store between. 
    Please give a detailed summary for positive, negative and neutral mixed feedbacks in March 2023, present them in bullet forms (At least 5 bullet points).
    What are the most of the critical issue of 9GAG overall.
    What do you suggest we focus on improving?
    """

    extract_filter_llm = OpenAI(temperature=0.0)
    info_dict = _extract_info_query(extract_filter_llm, q)
    print(f"info_dict extracted={info_dict}")

    filter_dict = _create_filter(infodict=info_dict)
    print(f"filter_dict created={filter_dict}")

    chat_llm = ChatOpenAI(temperature=0.5)
    chat_embeddings = HuggingFaceEmbeddings()
    _query_llm(index_name=index_name, filter_dict=filter_dict, embeddings=chat_embeddings, llm=chat_llm, query=q)

    # docs = _load_csv("./datafiles/reviews_202303.csv")
    # _upsert_pinecone(docs, embeddings)

    # docs = _load_csv("./datafiles/reviews_202304.csv")
    # _upsert_pinecone(docs, embeddings)

    # docs = _load_csv("./datafiles/reviews_202305.csv")
    # _upsert_pinecone(docs, embeddings)
    
    # begin = int(dateparser.parse("01-04-2023", date_formats=["%d-%m-%Y"]).timestamp() * 1000)
    # end = int(dateparser.parse("01-04-2023", date_formats=["%d-%m-%Y"]).timestamp() * 1000)

    # print(f"begin={begin}, end={end}")

    # 1646121600000.0
    # 1681116520708

    # docsearch = Pinecone.from_existing_index(index_name, embeddings)
    # res = docsearch.similarity_search("bad", filter={
    #     "$and": [
    #         { "last_update_ts": { "$gte": begin } }, 
    #         { "last_update_ts": { "$lte": end } }
    #     ]
    # })
    # res = docsearch.similarity_search("comment", filter={ "last_update_ts": { "$gte": begin } })
    # print(res)
    # print(_extract_dt("ss"))

    # _upsert_pinecone(docs, embeddings)

    # embeddings = OpenAIEmbeddings()
    # docsearch = Pinecone.from_existing_index(index_name, embeddings)
    # docsearch.similarity_search("")


    # llm = ChatOpenAI(temperature=0.5)
    # review_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), verbose=True)
    # q="""
    # You are a data analyst. The data you see are app reviews for the app called 9GAG from Google Play Store.  
    # Please give a detailed summary for positive, negative and neutral mixed feedbacks, present them in bullet forms (At least 5 bullet points).
    # What are the most of the critical issue of 9GAG overall apart from the recent comment changes, comment and "for you" section.
    # What do you suggest we focus on improving?
    # """
    # result = review_chain.run(q)
    # print(result)

    # docs = _load_csv("./datafiles/reviews_202303.csv")
    # q="""

    # docs = _load_csv("./datafiles/reviews_202303.csv")
    # print(docs[1])
    
    # docs = _load_csv("./datafiles/reviews_202304.csv")
    # embeddings = OpenAIEmbeddings()
    # _upsert_pinecone(docs, embeddings)
    # df = _load_df("./datafiles/reviews_202303.csv")
    # print(df.columns[0].title())
    # print(df.to_dict("records")[2])
    #_load_data("datafiles/reviews_202303.csv")
    #_load_data("datafiles/reviews_202303.csv")

if __name__ == "__main__":
    main()