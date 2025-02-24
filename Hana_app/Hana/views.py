# import io
# import sys
#
# from django.core.files.storage import default_storage
# from django.http import JsonResponse, HttpResponse
#
# import os
# from django.views.decorators.csrf import csrf_exempt
# from dotenv import load_dotenv
# from openai import OpenAI
#
# global connection_obj
# # db = MongoDBDatabase()
#
# # Configure OpenAI
#
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#
# client = OpenAI(api_key=OPENAI_API_KEY)
#
# USER = 'test_owner'
# PASSWORD = 'tcWI7unQ6REA'
# HOST = 'ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech:5432'
# DATABASE = 'test'
#
# #Text api for Cool_drinks_dataset.
# @csrf_exempt
# def gen_txt_response(request):
#     if request.method == "POST":
#         file = request.FILES.get('file')
#         table_name = file.name.split('.')[0]
#
#         # Define upload directory
#         upload_dir = os.path.join(os.getcwd(), 'upload')
#         os.makedirs(upload_dir, exist_ok=True)
#
#         # File path within upload directory
#         file_path = os.path.join(upload_dir, file.name)
#         with default_storage.open(file_path, 'wb+') as f:
#             for chunk in file.chunks():
#                 f.write(chunk)
#
#         df = file_to_sql(file_path, table_name, USER, PASSWORD, HOST, DATABASE)
#         print(df.head(5))
#
#         # Generate CSV metadata
#         csv_metadata = {"columns": df.columns.tolist()}
#         metadata_str = ", ".join(csv_metadata["columns"])
#         query = request.POST["query"]
#         prompt_eng = (
#             f"""
#             You are a Python expert focused on answering user queries about data preprocessing and analysis. Always strictly adhere to the following rules:
#
#             1. Data-Driven Queries:
#                 If the user's query is related to data processing or analysis, assume the `df` DataFrame in memory contains the actual uploaded data from the file "{file.name}" with the following columns: {metadata_str}.
#
#                 For such queries:
#                 - Generate Python code that directly interacts with the `df` DataFrame to provide accurate results strictly based on the data in the dataset.
#                 - Do not make any assumptions or provide any example outputs.
#                 - Ensure all answers are derived from actual calculations on the `df` DataFrame.
#                 - Include concise comments explaining key steps in the code.
#                 - Exclude any visualization, plotting, or assumptions about the data.
#
#                 Example:
#
#                 Query: "How many rows have 'Column1' > 100?"
#                 Response:
#                 ```python
#                 # Count rows where 'Column1' > 100
#                 count_rows = df[df['Column1'] > 100].shape[0]
#
#                 # Output the result
#                 print(count_rows)
#                 ```
#
#                 If the query asks for actionable items based on 'cleaned_comment', retrieve relevant rows containing keywords matching the query.
#
#                 Example:
#
#                 Query: "Get me the actionable items on Monster Energy cleaned comments"
#                 Response:
#                 ```python
#                 # Filter rows containing 'Monster Energy' in the 'cleaned_comment' column
#                 relevant_comments = df[df['cleaned_comment'].str.contains('Monster Energy', case=False, na=False)]
#
#                 # Output the relevant rows
#                 print(relevant_comments)
#                 ```
#
#             2. Invalid or Non-Data Queries:
#                 If the user's query is unrelated to data processing or analysis, or it cannot be answered using the dataset, respond with an appropriate print statement indicating the limitation. For example:
#
#                 Query: "What is AI?"
#                 Response:
#                 ```python
#                 print("This question is unrelated to the uploaded data. Please ask a data-specific query.")
#                 ```
#
#             3. Theoretical Concepts:
#                 If the user asks about theoretical concepts in data science or preprocessing (e.g., normalization, standardization), respond with a concise explanation. Keep the response focused and accurate.
#
#                 Example:
#
#                 Query: "What is normalization in data preprocessing?"
#                 Response:
#                 ```python
#                 print("Normalization is a data preprocessing technique used to scale numeric data within a specific range, typically [0, 1], to ensure all features contribute equally to the model.")
#                 ```
#
#             Remember:
#             - Always work with the actual data from the `df` DataFrame to generate responses.
#             - Never assume or provide sample results. All answers must be strictly derived from the dataset uploaded by the user.
#             - Respond with Python code or appropriate concise print statements as described above.
#
#             User query: {query}.
#             """
#         )
#         code = generate_code(prompt_eng)
#         # Execute the generated code
#         result = execute_py_code(code, df)
#         return JsonResponse({"answer": markdown_to_html(result)})
#     return HttpResponse("Invalid Request Method", status=405)
#
# import markdown
# def markdown_to_html(md_text):
#     html_text = markdown.markdown(md_text)
#     return html_text
#
#
# # Function to generate code from OpenAI API
# def generate_code(prompt_eng):
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt_eng}
#         ]
#     )
#     all_text = ""
#     for choice in response.choices:
#         message = choice.message
#         chunk_message = message.content if message else ''
#         all_text += chunk_message
#     print(all_text)
#     if "```python" in all_text:
#         code_start = all_text.find("```python") + 9
#         code_end = all_text.find("```", code_start)
#         code = all_text[code_start:code_end]
#     else:
#         code = all_text
#     return code
#
#
# def execute_py_code(code, df):
#     # Create a string buffer to capture the output
#     buffer = io.StringIO()
#     sys.stdout = buffer
#
#     # Create a local namespace for execution
#     local_vars = {'df': df}
#
#     try:
#         # Execute the code
#         exec(code, globals(), local_vars)
#
#         # Get the captured output
#         output = buffer.getvalue().strip()
#
#         # If there's no output, try to get the last evaluated expression
#         if not output:
#             last_line = code.strip().split('\n')[-1]
#             if not last_line.startswith(('print', 'return')):
#                 output = eval(last_line, globals(), local_vars)
#                 print(output)
#     except Exception as e:
#         output = f"Error executing code: {str(e)}"
#     finally:
#         # Reset stdout
#         sys.stdout = sys.__stdout__
#
#     return str(output)
#
#
# def execute_query(query, user, password, host, db_name):
#     from sqlalchemy import create_engine, text
#
#     engine = create_mysql_engine(user, password, host, db_name)
#     with engine.connect() as connection:
#         try:
#             result_set = connection.execute(text(query))
#             output = []
#             for row in result_set:
#                 print(row)
#                 output.append(str(row))
#             return output
#         except Exception as e:
#             return str(e)
#
#
# # @csrf_exempt
# # def gen_graph_response(request):
# #     if request.method == "POST":
# #         file = request.FILES.get('file')
# #         table_name = file.name.split('.')[0]
# #
# #         # Define upload directory
# #         upload_dir = os.path.join(os.getcwd(), 'upload')
# #         os.makedirs(upload_dir, exist_ok=True)
# #
# #         # File path within upload directory
# #         file_path = os.path.join(upload_dir, file.name)
# #         with default_storage.open(file_path, 'wb+') as f:
# #             for chunk in file.chunks():
# #                 f.write(chunk)
# #
# #         df = file_to_sql(file_path, table_name, USER, PASSWORD, HOST, DATABASE)
# #         print(df.head(5))
# #
# #         # Generate CSV metadata
# #         csv_metadata = {"columns": df.columns.tolist()}
# #         metadata_str = ", ".join(csv_metadata["columns"])
# #         query = request.POST["query"]
# #
# #         # Updated prompt to incorporate `df`
# #         prompt_eng = (
# #             f"You are an AI specialized in data analytics and visualization."
# #             f"Data used for analysis is stored in a pandas DataFrame named `df`. "
# #             f"The DataFrame `df` contains the following attributes: {metadata_str}. "
# #             f"Based on the user's query, generate Python code using Plotly to create the requested type of graph "
# #             f"(e.g., bar, pie, scatter, etc.) using the data in the DataFrame `df`. "
# #             f"The graph must utilize the data within `df` as appropriate for the query. "
# #             f"If the user does not specify a graph type, decide whether to generate a line or bar graph based on the situation. "
# #             f"Every graph must include a title, axis labels (if applicable), and appropriate colors for better visualization. "
# #             f"The graph must have a white background for both the plot and paper. "
# #             f"The code must output a Plotly 'Figure' object stored in a variable named 'fig', "
# #             f"and include 'data' and 'layout' dictionaries compatible with React. "
# #             f"The user asks: {query}."
# #         )
# #
# #         # Call AI to generate the code
# #         chat = generate_code(prompt_eng)
# #         print("Generated code from AI:")
# #         print(chat)
# #
# #         # Check for valid Plotly code in the AI response
# #         if 'import' in chat:
# #             namespace = {'df': df}  # Pass `df` into the namespace
# #             try:
# #                 # Execute the generated code
# #                 exec(chat, namespace)
# #
# #                 # Retrieve the Plotly figure from the namespace
# #                 fig = namespace.get("fig")
# #
# #                 if fig and isinstance(fig, Figure):
# #                     # Convert the Plotly figure to JSON
# #                     chart_data = fig.to_plotly_json()
# #
# #                     # Ensure JSON serialization by converting NumPy arrays to lists
# #                     def make_serializable(obj):
# #                         if isinstance(obj, np.ndarray):
# #                             return obj.tolist()
# #                         elif isinstance(obj, dict):
# #                             return {k: make_serializable(v) for k, v in obj.items()}
# #                         elif isinstance(obj, list):
# #                             return [make_serializable(v) for v in obj]
# #                         return obj
# #
# #                     # Recursively process the chart_data
# #                     chart_data_serializable = make_serializable(chart_data)
# #
# #                     # Return the structured response to the frontend
# #                     return JsonResponse({
# #                         "chartData": chart_data_serializable
# #                     }, status=200)
# #                 else:
# #                     print("No valid Plotly figure found.")
# #                     return JsonResponse({"message": "No valid Plotly figure found."}, status=200)
# #             except Exception as e:
# #                 error_message = f"There was an error while executing the code: {str(e)}"
# #                 print(error_message)
# #                 return JsonResponse({"message": error_message}, status=500)
# #         else:
# #             print("Invalid AI response.")
# #             return JsonResponse({"message": "AI response does not contain valid code."}, status=400)
# #
# #     # Return a fallback HttpResponse for invalid request methods
# #     return HttpResponse("Invalid request method", status=405)
# #
# #
# def file_to_sql(file_path, table_name, user, password, host, db_name):
#     import pandas as pd
#     import os
#     from sqlalchemy import create_engine
#
#     # engine = create_engine(f"postgresql://{user}:{password}@{host}/{db_name}")
#     engine = create_mysql_engine(user, password, host, db_name)
#
#     if not table_name:
#         table_name = os.path.splitext(os.path.basename(file_path))[0]
#
#     file_extension = os.path.splitext(file_path)[-1].lower()
#     if file_extension == '.xlsx':
#         df = pd.read_excel(file_path)
#     elif file_extension == '.csv':
#         df = pd.read_csv(file_path)
#     else:
#         raise ValueError("Unsupported file format. Please provide an Excel (.xlsx) or CSV (.csv) file.")
#
#     df.to_sql(table_name, con=engine, if_exists='replace', index=False)
#     return df
#
#
# def create_mysql_engine(user, password, host, db_name):
#     from sqlalchemy import create_engine, text
#
#     if db_name:
#         connection_str = f'postgresql://{user}:{password}@{host}/{db_name}'
#     else:
#         connection_str = f'postgresql://{user}:{password}@{host}/'
#     engine = create_engine(connection_str)
#     return engine
#
# #
# # #Both text and graph.
# # @csrf_exempt
# # def gen_response(request):
# #     if request.method == "POST":
# #         # Check if a file has been uploaded
# #         file = request.FILES.get('file')
# #         table_name = file.name.split('.')[0]
# #
# #         # Define upload directory
# #         upload_dir = os.path.join(os.getcwd(), 'upload')
# #         os.makedirs(upload_dir, exist_ok=True)
# #
# #         # File path within upload directory
# #         file_path = os.path.join(upload_dir, file.name)
# #         with default_storage.open(file_path, 'wb+') as f:
# #             for chunk in file.chunks():
# #                 f.write(chunk)
# #
# #         df = file_to_sql(file_path, table_name, USER, PASSWORD, HOST, DATABASE)
# #         print(df.head(5))
# #
# #         # Generate CSV metadata
# #         csv_metadata = {"columns": df.columns.tolist()}
# #         metadata_str = ", ".join(csv_metadata["columns"])
# #         query = request.POST["query"]
# #         if not query:
# #             return JsonResponse({"error": "No query provided"}, status=400)
# #
# #         graph_keywords = [
# #             "plot", "graph", "visualize", "visualization", "scatter", "bar chart",
# #             "line chart", "histogram", "pie chart", "bubble chart", "heatmap", "box plot",
# #             "generate chart", "create graph", "draw", "trend", "correlation"
# #         ]
# #
# #         # Decide whether the query is text-based or graph-based
# #         if any(keyword in query.lower() for keyword in graph_keywords):
# #             # Graph-related prompt
# #             print("if_condition------------------")
# #             prompt_eng = (
# #                 f"You are an AI specialized in data analytics and visualization."
# #                 f"Data used for analysis is stored in a pandas DataFrame named `df`. "
# #                 f"The DataFrame `df` contains the following attributes: {metadata_str}. "
# #                 f"Based on the user's query, generate Python code using Plotly to create the requested type of graph "
# #                 f"(e.g., bar, pie, scatter, etc.) using the data in the DataFrame `df`. "
# #                 f"The graph must utilize the data within `df` as appropriate for the query. "
# #                 f"If the user does not specify a graph type, decide whether to generate a line or bar graph based on the situation."
# #                 f"Every graph must include a title, axis labels (if applicable), and appropriate colors for better visualization."
# #                 f"The graph must have a white background for both the plot and paper. "
# #                 f"The code must output a Plotly 'Figure' object stored in a variable named 'fig', "
# #                 f"and include 'data' and 'layout' dictionaries compatible with React. "
# #                 f"The user asks: {query}."
# #             )
# #
# #             # Call AI to generate the code
# #             chat = generate_code(prompt_eng)
# #             print("Generated code from AI:")
# #             print(chat)
# #
# #             # Check for valid Plotly code in the AI response
# #             if 'import' in chat:
# #                 namespace = {'df': df}  # Pass `df` into the namespace
# #                 try:
# #                     # Execute the generated code
# #                     exec(chat, namespace)
# #
# #                     # Retrieve the Plotly figure from the namespace
# #                     fig = namespace.get("fig")
# #
# #                     if fig and isinstance(fig, Figure):
# #                         # Convert the Plotly figure to JSON
# #                         chart_data = fig.to_plotly_json()
# #
# #                         # Ensure JSON serialization by converting NumPy arrays to lists
# #                         def make_serializable(obj):
# #                             if isinstance(obj, np.ndarray):
# #                                 return obj.tolist()
# #                             elif isinstance(obj, dict):
# #                                 return {k: make_serializable(v) for k, v in obj.items()}
# #                             elif isinstance(obj, list):
# #                                 return [make_serializable(v) for v in obj]
# #                             return obj
# #
# #                         # Recursively process the chart_data
# #                         chart_data_serializable = make_serializable(chart_data)
# #
# #                         # Return the structured response to the frontend
# #                         return JsonResponse({
# #                             "chartData": chart_data_serializable
# #                         }, status=200)
# #                     else:
# #                         print("No valid Plotly figure found.")
# #                         return JsonResponse({"message": "No valid Plotly figure found."}, status=200)
# #                 except Exception as e:
# #                     error_message = f"There was an error while executing the code: {str(e)}"
# #                     print(error_message)
# #                     return JsonResponse({"message": error_message}, status=500)
# #             else:
# #                 print("Invalid AI response.")
# #                 return JsonResponse({"message": "AI response does not contain valid code."}, status=400)
# #
# #
# #         else:
# #             # Text-based prompt
# #             print("else_condition------------------")
# #             prompt_eng = (
# #                 f"""
# #                     You are a Python expert focused on answering user queries about data preprocessing and analysis. Always strictly adhere to the following rules:
# #
# #                     1. Data-Driven Queries:
# #                         If the user's query is related to data processing or analysis, assume the `df` DataFrame in memory contains the actual uploaded data from the file "{file.name}" with the following columns: {metadata_str}.
# #
# #                         For such queries:
# #                         - Generate Python code that directly interacts with the `df` DataFrame to provide accurate results strictly based on the data in the dataset.
# #                         - Do not make any assumptions or provide any example outputs.
# #                         - Ensure all answers are derived from actual calculations on the `df` DataFrame.
# #                         - Include concise comments explaining key steps in the code.
# #                         - Exclude any visualization, plotting, or assumptions about the data.
# #
# #                         Example:
# #
# #                         Query: "How many rows have 'Column1' > 100?"
# #                         Response:
# #                         ```python
# #                         # Count rows where 'Column1' > 100
# #                         count_rows = df[df['Column1'] > 100].shape[0]
# #
# #                         # Output the result
# #                         print(count_rows)
# #                         ```
# #
# #                     2. Invalid or Non-Data Queries:
# #                         If the user's query is unrelated to data processing or analysis, or it cannot be answered using the dataset, respond with an appropriate print statement indicating the limitation. For example:
# #
# #                         Query: "What is AI?"
# #                         Response:
# #                         ```python
# #                         print("This question is unrelated to the uploaded data. Please ask a data-specific query.")
# #                         ```
# #
# #                     3. Theoretical Concepts:
# #                         If the user asks about theoretical concepts in data science or preprocessing (e.g., normalization, standardization), respond with a concise explanation. Keep the response focused and accurate.
# #
# #                         Example:
# #
# #                         Query: "What is normalization in data preprocessing?"
# #                         Response:
# #                         ```python
# #                         print("Normalization is a data preprocessing technique used to scale numeric data within a specific range, typically [0, 1], to ensure all features contribute equally to the model.")
# #                         ```
# #
# #                     User query: {query}.
# #                 """
# #             )
# #
# #             # Generate text-related code
# #             code = generate_code(prompt_eng)
# #             print("Generated code from AI (Text):")
# #             print(code)
# #
# #             # Execute the generated code with the dataset
# #             result = execute_py_code(code, df)
# #             return JsonResponse({"answer": result}, status=200)
# #
# #     return HttpResponse("Invalid request method", status=405)


import os
import io
import sys
import markdown
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from django.core.files.storage import default_storage
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

USER = 'test_owner'
PASSWORD = 'tcWI7unQ6REA'
HOST = 'ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech:5432'
DATABASE = 'test'


@csrf_exempt
def gen_txt_response(request):
    if request.method == "POST":
        query = request.POST["query"].lower()
        # Handle user greetings with LLM response
        greetings = {"hi", "hello", "hey", "greetings"}
        if query in greetings:
            greeting_prompt = "Respond to the user greeting in a friendly and engaging manner."
            greeting_response = generate_code(greeting_prompt)
            return JsonResponse({"answer": markdown_to_html(greeting_response)})
        # elif query:
        #     # Handle general topic-based queries with LLM response
        #     topic_prompt = f"Provide an informative response about '{query}' in a structured manner."
        #     topic_response = generate_code(topic_prompt)
        #     return JsonResponse({"answer": markdown_to_html(topic_response)})
        else:
            file = request.FILES.get('file')
            table_name = file.name.split('.')[0]

            # Define upload directory
            upload_dir = os.path.join(os.getcwd(), 'upload')
            os.makedirs(upload_dir, exist_ok=True)

            # File path within upload directory
            file_path = os.path.join(upload_dir, file.name)
            with default_storage.open(file_path, 'wb+') as f:
                for chunk in file.chunks():
                    f.write(chunk)

            df = file_to_sql(file_path, table_name, USER, PASSWORD, HOST, DATABASE)
            # Extract only required columns and drop NaN values
            df = df[['cleaned_comment', 'sentiment_category', 'category', 'subcategory']].dropna()
            print(df.shape)

            # Keep only negative and neutral sentiment categories
            df = df[df['sentiment_category'].isin(['negative', 'neutral'])]
            print(df.head(4))

            prompt_eng = f"""
            Based on the following comments related to '{query}', provide structured, actionable insights.
            Identify key themes from {df['category']} and {df['subcategory']} relevant to the user's query.
            Specifically focus on keywords mentioned in the query (e.g., "shit
            ," "fall sick") and derive insights to address or mitigate negative sentiment.Don't mention the  words like 'shit' and 'fallsick' in the final output.
            The insights should be structured with bullet points and categorized appropriately based on the nature of the comments.
            Do not include the original comments; only provide actionable recommendations.
            Ensure that the categories are dynamically generated based on the context of the query rather than using predefined topics.
            The dataset is provided in the dataframe as 'df'.
            Always You can just give the headings and the information related to the headings based on the {query} and {df}.
            You Strictly use the blue colour for the heading for the Final Answer.
            Make sure that If you got any comparison related terms in the {query},then compare the issues regarding the comparable items given and produce the final output with clear comparable statements based on the {df['cleaned_comment']}. For example,if you give a comparison between monster energy and red bull,,then you can make the comparable differences with the red bull and monster energy and generate actionable insights based on {query}.
            In Comparison,the final output should be like this:
            Compare both the things.
            # Monster Energy heavily sponsors extreme sports events, athletes, and teams, establishing a strong presence in this niche.
            # Red Bull also invests in extreme sports but has a broader focus, including lifestyle and mainstream sports, potentially reaching a wider audience.
            
            Do not include this in the final output:
            'Based on the analysis of the comments related to Monster Energy, here are structured, actionable insights derived from the identified themes and keywords associated with negative sentiments, particularly focusing on terms including "shit" and "sick."'
            
            Note: Strictly Do not use the above example at the time of  every query procesinng.The above is just an example. Actual output should always be dynamically generated based on the query.
                  Ensure the response is not a repeated or templated output but is always fresh and query-specific.
                  Do not include the general sentiment analysis like neutral feedback,Negative feedback etc in the final output.
                 
    
            """
            insights = generate_code(prompt_eng)
            print(insights)
            structured_response = (
                "Here are some actionable insights based on your query:\n\n"
                f"{insights}"
            )
            return JsonResponse({"answer": markdown_to_html(structured_response)})

    return HttpResponse("Invalid Request Method", status=405)


import markdown
from bs4 import BeautifulSoup
def markdown_to_html(md_text):
    # Convert Markdown to HTML
    html_content = markdown.markdown(md_text)

    # Parse the HTML to modify headings
    soup = BeautifulSoup(html_content, "html.parser")

    # Apply blue color to all heading tags
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        tag['style'] = "color: blue;"

    return str(soup)


def generate_code(prompt_eng):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing actionable insights."},
            {"role": "user", "content": prompt_eng}
        ]
    )
    return response.choices[0].message.content.strip()


def file_to_sql(file_path, table_name, user, password, host, db_name):
    import pandas as pd
    import os
    from sqlalchemy import create_engine

    # engine = create_engine(f"postgresql://{user}:{password}@{host}/{db_name}")
    engine = create_mysql_engine(user, password, host, db_name)

    if not table_name:
        table_name = os.path.splitext(os.path.basename(file_path))[0]

    file_extension = os.path.splitext(file_path)[-1].lower()
    if file_extension == '.xlsx':
        df = pd.read_excel(file_path)
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide an Excel (.xlsx) or CSV (.csv) file.")

    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    return df


def create_mysql_engine(user, password, host, db_name):
    from sqlalchemy import create_engine, text

    if db_name:
        connection_str = f'postgresql://{user}:{password}@{host}/{db_name}'
    else:
        connection_str = f'postgresql://{user}:{password}@{host}/'
    engine = create_engine(connection_str)
    return engine
