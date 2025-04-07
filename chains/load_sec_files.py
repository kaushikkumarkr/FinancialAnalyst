from retrievers.sec_vector_store import load_and_split_sec_file, build_vectorstore
import os

# def embed_sec_folder(folder_path):
#     all_docs = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             path = os.path.join(folder_path, filename)
#             metadata = {
#                 "company": filename.split("_")[0],
#                 "filing_year": filename.split("_")[1].replace(".txt", ""),
#                 "filing_type": "10-K"
#             }
#             chunks = load_and_split_sec_file(path, metadata)
#             all_docs.extend(chunks)

#     build_vectorstore(all_docs)
#     print("✅ All filings embedded!")



import re

def embed_sec_folder(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            path = os.path.join(folder_path, filename)
            
            # Example filename format: "Tesla_2022_10-K.txt"
            match = re.match(r"(.+?)_(\d{4})_(10-K|10-Q|8-K)\.txt", filename)
            if not match:
                print(f"⚠️ Skipping unrecognized format: {filename}")
                continue

            company, year, filing_type = match.groups()

            metadata = {
                "company": company,
                "filing_year": year,
                "filing_type": filing_type
            }

            chunks = load_and_split_sec_file(path, metadata)
            all_docs.extend(chunks)

    build_vectorstore(all_docs)
    print("✅ All filings embedded!")


if __name__ == "__main__":
    embed_sec_folder("sec_data/")
