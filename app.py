# import streamlit as st
# import base64
# import pandas as pd
# import numpy
# import re 
# import ast
# from groq import Groq
# import matplotlib.pyplot as plt
# st.title("ChatGPT-like clone")

# #API

# client = Groq(
#     api_key="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",
# )
# df = []
# tst = ""
# info = ""
# cnt=1


# #search dataset

# def search_dataset(message) :
#     # Search DATASET ...

#     chat_completion = client.chat.completions.create(
#     messages=[

#         {
#             "role": "system",
#             "content": "You are a Professional Data Scientist , Search for top 5 links from kaggle to download the dataset requested by the user , get some somewhere else if it is not in Kaggle , be precise" ,
#         },
#         {
#             "role": "user",
#             "content": f"{message}" ,
#         }
#     ],
#     model="llama3-70b-8192",
#     )
#     response_text = chat_completion.choices[0].message.content
#     return response_text

# def preprocess(mess, tst):
#     # Use the 'tst' parameter which is the dataset loaded into a variable named 'df'

#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a Professional Data Scientist. Write a Python code to complete the requested task, do all inplace so that it updates in real-time.",
#             },
#             {
#                 "role": "user",
#                 "content": f"This is the dataset I have: {tst}. Give a code to {mess}, also call it. Give code only, don't give explanation.",
#             }
#         ],
#         model="llama3-70b-8192",
#     )

#     viss = chat_completion.choices[0].message.content
#     viss = re.sub('python', '', viss, flags=re.IGNORECASE)
#     viss = re.sub('Python', '', viss, flags=re.IGNORECASE)

#     pattern = r"```(.*?)```"
#     matches = re.findall(pattern, viss, re.DOTALL)

#     if matches:
#         try:
           
#             parsed_ast = ast.parse(matches[0])
#             local_context = {}
#             exec(compile(parsed_ast, filename="<ast>", mode="exec"), globals(), local_context)
#             result = local_context.get('result', "Execution completed.")
            
#             return result
#         except Exception as e:
#             return f"Error executing code: {str(e)}"
#     else:
#         return "No valid Python code found in the response."
    
# def process_file(file):
#     global df, tst, info
    
#     if file is not None:
#         # Save the uploaded file
#         with open('data.csv', 'wb') as f:
#             f.write(file.getvalue())
        
#         # Read the CSV file into a DataFrame
#         df = pd.read_csv('data.csv')
        
#         # Store dataset head and info
#         tst = df.head()
#         info = df.info()

#         # Use Groq to generate an overview message
#         chat_completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": f"This is the dataset I have: {tst}, explain and overview of it in 2 lines",
#                 }
#             ],
#             model="llama3-70b-8192",
#         )
        
#         response_text = chat_completion.choices[0].message.content
#         return response_text
    
#     else:
#         return 'No file provided'
    
# def suggest() :
#     # Suggessions

#     chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": f"I'm working on preprocessing a DataSet ,  This is the dataset i have : {tst} and its info : {info},give me 3 suggests to what to do further in 3 lines" ,
#         }
#     ],
#     model="llama3-70b-8192",
#     )
#     response_text = chat_completion.choices[0].message.content
#     return response_text

# def others(mess) :

#     chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": f"You are a AI data analyst , answer the below question according to your role , QUESTION : {mess}" ,
#         }
#     ],
#     model="llama3-70b-8192",
#     )
#     response_text = chat_completion.choices[0].message.content
#     return response_text

# def visualize(mess):
#     global cnt
#     global df, tst
    
#     if df is None:
#         return "Please upload a dataset first."
    
#     # Use Groq to generate code for visualizing the specified columns
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"This is the dataset I have: {tst}, give a code to visualize {mess} using matplotlib, also call it, give the column names exactly as in the dataset, and save the plot as 'plot{cnt}.png'."
#             }
#         ],
#         model="llama3-70b-8192",
#     )
    
#     viss = chat_completion.choices[0].message.content
#     viss = re.sub('python', '', viss, flags=re.IGNORECASE)
#     viss = re.sub('Python', '', viss, flags=re.IGNORECASE)
    
#     pattern = r"```(.*?)```"
#     matches = re.findall(pattern, viss, re.DOTALL)
    
#     if not matches:
#         return "Visualization code not generated."
    
#     # Safely parse the response text as a Python AST (Abstract Syntax Tree)
#     try:
#         parsed_ast = ast.parse(matches[0])
#     except SyntaxError:
#         return "Error parsing visualization code."

#     # Execute the parsed AST
#     try:
#         exec(compile(parsed_ast, filename="<ast>", mode="exec"))
#     except Exception as e:
#         return f"Error executing visualization code: {str(e)}"

#     # Save the plot
#     plot_path = f"plot{cnt}.png"
#     plt.savefig(plot_path)
#     cnt += 1
    
#     return plot_path

# uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
# if st.button('Process File') and uploaded_file:
#     result = process_file(uploaded_file)
#     st.write(f"Response: {result}")
# if prompt := st.chat_input("What is up?"):
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         st.markdown(prompt)

import streamlit as st
import pandas as pd
from groq import Groq
import re
import ast
import requests
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie 
from io import BytesIO
import io
import base64
from contextlib import redirect_stdout
from autogluon.tabular import TabularDataset, TabularPredictor

# Initialize Groq client
client = Groq(api_key="")

# Global variables
df = none
tst = None
info = None
cnt = 1  

st.markdown(
    """
    <style>
    .cover-glow {
        width: 100%;
        height: auto;
        padding: 3px;
        box-shadow: 
            0 0 5px #330000,
            0 0 10px #660000,
            0 0 15px #990000,
            0 0 20px #CC0000,
            0 0 25px #FF0000,
            0 0 30px #FF3333,
            0 0 35px #FF6666;
        position: relative;
        z-index: -1;
        border-radius: 30px;  /* Rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_path = r"E:\project\DataGenie-main\DataGenie-main\logo.png"
img_base64 = img_to_base64(img_path)
st.sidebar.markdown(
    f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

left_column, right_column = st.columns((2,4))

with left_column:
    logo='https://lottie.host/d0aac6b8-9fdb-4f33-89f9-4cd0ead97f56/4fbpP4dtXM.json'
    logo_image=load_lottieurl(logo)
    st_lottie(logo_image,width=300,height=100,key='logo')
with right_column:
    st.title("Data Science Assistant")

def search_dataset(message):
    # Implement search dataset functionality using Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a Professional Data Scientist. Search for top 5 links from Kaggle to download the dataset requested by the user. Get some elsewhere if it is not on Kaggle. Be precise.",
            },
            {
                "role": "user",
                "content": f"{message}",
            }
        ],
        model="llama3-70b-8192",
    )
    response_text = chat_completion.choices[0].message.content
    return response_text



def preprocess(mess):
    if df is None:
        return "Please upload a dataset first."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a Professional Data Scientist. Write a python code to complete the requested task, do all in place so that it updates in real-time.",
            },
            {
                "role": "user",
                "content": f"This is the dataset I have: {tst} and it is loaded in a variable named as df. Give a code to {mess}, also call it, give code only don't give explanation.",
            }
        ],
        model="llama3-70b-8192",
    )

    viss = chat_completion.choices[0].message.content
    viss = re.sub('python', '', viss, flags=re.IGNORECASE)
    viss = re.sub('Python', '', viss, flags=re.IGNORECASE)

    pattern = r"```(.*?)```"
    matches = re.findall(pattern, viss, re.DOTALL)

    if not matches:
        return "Code snippet not generated."

    # Safely parse the response text as a Python AST (Abstract Syntax Tree)
    try:
        parsed_ast = ast.parse(matches[0])
    except SyntaxError as e:
        return f"Error parsing code: {str(e)}"

    # Capture the output of the exec
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            # Wrap the exec statement to capture and format pandas output
            exec(
                compile(parsed_ast, filename="<ast>", mode="exec"),
                globals()
            )
        except Exception as e:
            return f"Error during execution: {str(e)}"

    # Get the output
    output = f.getvalue()

    # Ensure proper structure and formatting for pandas DataFrames/Series
    if 'pd' in globals():
        try:
            if isinstance(output, pd.DataFrame) or isinstance(output, pd.Series):
                output = output.to_string()
        except Exception:
            pass

    return output if output else "No output generated."

def visualize(mess):
    global cnt, df

    if df is None:
        return "Please upload a dataset first."

    # Use Groq to generate code for visualizing the specified columns
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"This is the dataset I have: {tst}, give a code to visualize {mess} using matplotlib, also call it, give the column names exactly as in the dataset, and save the plot as 'plot{cnt}.png'."
            }
        ],
        model="llama3-70b-8192",
    )

    viss = chat_completion.choices[0].message.content
    viss = re.sub('python', '', viss, flags=re.IGNORECASE)
    viss = re.sub('Python', '', viss, flags=re.IGNORECASE)

    pattern = r"```(.*?)```"
    matches = re.findall(pattern, viss, re.DOTALL)

    if not matches:
        return "Visualization code not generated."

    # Safely parse the response text as a Python AST (Abstract Syntax Tree)
    try:
        parsed_ast = ast.parse(matches[0])
    except SyntaxError as e:
        return f"Error parsing visualization code: {str(e)}"

    # Execute the parsed AST
    try:
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))
    except Exception as e:
        return f"Error executing visualization code: {str(e)}"

    # Save the plot
    plot_path = f"plot{cnt}.png"
    plt.savefig(plot_path)
    cnt += 1

    return plot_path


def suggest():
    if df is None:
        return "Please upload a dataset first."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"I'm working on preprocessing a DataSet. This is the dataset I have: {tst} and its info: {info}. Give me 3 suggestions on what to do further in 3 lines."
            }
        ],
        model="llama3-70b-8192",
    )
    response_text = chat_completion.choices[0].message.content
    return response_text


def model_leaderboard(train_df):
    predictor = TabularPredictor(
        label='Anaemic',
        eval_metric='accuracy'
    ).fit(
        train_df,
        presets=['best_quality']
    )
    leaderboard = predictor.leaderboard(train_df, silent=True)
    return leaderboard


def others(mess):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"You are an AI data analyst. Answer the below question according to your role. QUESTION: {mess}",
            }
        ],
        model="llama3-70b-8192",
    )
    response_text = chat_completion.choices[0].message.content
    return response_text


def classify(mess):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an assistant with knowledge on Data Science who classifies user prompt into one of the following categories:
              visualization: if user needs to visualize the data using any plots or charts,
              searching: If the user is searching for a dataset available online or wants any dataset to be searched,
              data_cleaning: If the user wants to do any cleaning or preprocessing tasks,
              model_training: If the user wants to train a Model,
              suggestions: if the user wants to get some suggestions on what to do with the dataset,
              Quering_dataset: If the user wants to check what is in the dataset like unique values, data types, df.head(), or df.tail(),
              Others: If the query doesn't fall into any of the above classes like Hi, Hello, and more.

              \n\n Don't give any kind of Explanation just give the classification """,
            },
            {
                "role": "user",
                "content": f"{mess}",
            }
        ],
        model="llama3-70b-8192",
    )
    response_text = chat_completion.choices[0].message.content
    return response_text


def process_message(message):
    cls = classify(message)

    if "visualization" in cls:
        return visualize(message)
    elif "searching" in cls:
        return search_dataset(message)
    elif 'data_cleaning' in cls or 'Quering_dataset' in cls:
        return preprocess(message)
    elif 'suggestions' in cls:
        return suggest()
    elif 'model_training' in cls:
        return model_leaderboard(df) if df is not None else "Please upload a dataset first."
    else:
        return others(message)


# Streamlit UI code
def main():

    # Sidebar for uploading dataset
    st.sidebar.title("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        st.sidebar.markdown("**File uploaded!**")
        # Process the uploaded file
        file_result = process_file(uploaded_file)
        st.sidebar.write(file_result)
        st.sidebar.write(df.head())  # Display top 5 data points

    # Chat interface for user query
    if prompt := st.chat_input("Enter your query:"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            result = process_message(prompt)
            if isinstance(result, str) and result.endswith(".png"):
                st.image(result, use_column_width=True)
            elif isinstance(result, pd.DataFrame):
                st.dataframe(result)
            else:
                st.write(result)

    # Download button
    if df is not None:
        csv_string = df.to_csv(index=False)
        csv_bytes = csv_string.encode()
        st.sidebar.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name="data.csv",
            mime="text/csv"
        )


def process_file(file):
    global df, tst, info
    
    if file is not None:
        # Save the uploaded file
        with open('data.csv', 'wb') as f:
            f.write(file.getvalue())
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv('data.csv')
        
        # Store dataset head and info
        tst = df.head()

        # Capture df.info() output
        info = df.info()
        # Use Groq to generate an overview message
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"This is the dataset I have: {tst}, explain and overview of it in 2 lines",
                }
            ],
            model="llama3-70b-8192",
        )
        
        response_text = chat_completion.choices[0].message.content
        return response_text
    
    else:
        return 'No file provided'


if __name__ == "__main__":
    main()
