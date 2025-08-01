import pinecone
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class VectorDatabase:
    def __init__(self, api_key, environment="us-east-1"):
        """
        Initialize the VectorDatabase with Pinecone API key and environment.
        
        Args:
            api_key (str): Pinecone API key
            environment (str): Pinecone environment
        """
        self.api_key = api_key
        self.environment = environment
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "product-recommendations"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_index(self, dimension=384):
        """
        Create a Pinecone index for storing product vectors.
        
        Args:
            dimension (int): Vector dimension (default 384 for all-MiniLM-L6-v2)
        """
        try:
            # Check if index already exists
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                print(f"Index '{self.index_name}' created successfully.")
            else:
                print(f"Index '{self.index_name}' already exists.")
        except Exception as e:
            print(f"Error creating index: {e}")
    
    def get_index(self):
        """
        Get the Pinecone index instance.
        
        Returns:
            pinecone.Index: The Pinecone index instance
        """
        return self.pc.Index(self.index_name)
    
    def vectorize_products(self, df):
        """
        Convert product descriptions to vectors using SentenceTransformer.
        
        Args:
            df (pandas.DataFrame): DataFrame containing product data
            
        Returns:
            list: List of vectors
        """
        descriptions = df['Description'].tolist()
        vectors = self.model.encode(descriptions)
        return vectors.tolist()
    
    def upsert_products(self, df):
        """
        Insert or update product vectors in the Pinecone index.
        
        Args:
            df (pandas.DataFrame): DataFrame containing product data
        """
        try:
            index = self.get_index()
            vectors = self.vectorize_products(df)
            
            # Prepare data for upsert
            upsert_data = []
            for i, (_, row) in enumerate(df.iterrows()):
                metadata = {
                    'stock_code': str(row['StockCode']),
                    'description': str(row['Description']),
                    'unit_price': float(row['UnitPrice']),
                    'country': str(row['Country'])
                }
                upsert_data.append((
                    f"product_{row['StockCode']}_{i}",
                    vectors[i],
                    metadata
                ))
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i:i + batch_size]
                index.upsert(vectors=batch)
            
            print(f"Successfully upserted {len(upsert_data)} products.")
        except Exception as e:
            print(f"Error upserting products: {e}")
    
    def search_similar_products(self, query, top_k=5):
        """
        Search for similar products based on a natural language query.
        
        Args:
            query (str): Natural language query
            top_k (int): Number of top results to return
            
        Returns:
            list: List of similar products with metadata
        """
        try:
            index = self.get_index()
            query_vector = self.model.encode([query]).tolist()[0]
            
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            products = []
            for match in results['matches']:
                product = {
                    'id': match['id'],
                    'score': match['score'],
                    'stock_code': match['metadata']['stock_code'],
                    'description': match['metadata']['description'],
                    'unit_price': match['metadata']['unit_price'],
                    'country': match['metadata']['country']
                }
                products.append(product)
            
            return products
        except Exception as e:
            print(f"Error searching products: {e}")
            return []
    
    def generate_response(self, query, products):
        """
        Generate a natural language response based on the query and found products.
        
        Args:
            query (str): Original query
            products (list): List of found products
            
        Returns:
            str: Natural language response
        """
        if not products:
            return "I'm sorry, I couldn't find any products matching your query. Please try with different keywords."
        
        response = f"Based on your query '{query}', I found {len(products)} relevant products:\n\n"
        
        for i, product in enumerate(products, 1):
            response += f"{i}. {product['description']} - ${product['unit_price']:.2f}\n"
        
        response += "\nThese products seem to match what you're looking for. Would you like more details about any of them?"
        
        return response

if __name__ == '__main__':
    # Example usage
    api_key = "pcsk_tADBr_AhAdNdaGjsR7WhHvQZcDvThAqvCvVmkRUhLPc87QmXcNGmco8dUM9MBJausYV1X"
    vdb = VectorDatabase(api_key)
    vdb.create_index()

