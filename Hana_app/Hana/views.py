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
#         # Generate CSV metadata
#         csv_metadata = {"columns": df.columns.tolist()}
#         metadata_str = ", ".join(csv_metadata["columns"])
#         query = request.POST["query"]
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
#             prompt_eng = (
#                 f"""
#                     You are a Python expert focused on answering user queries about data preprocessing and analysis. Always strictly adhere to the following rules:
#
#                     1. Data-Driven Queries:
#                         If the user's query is related to data processing or analysis, assume the `df` DataFrame in memory contains the actual uploaded data from the file "{file.name}" with the following columns: {metadata_str}.
#
#                         For such queries:
#                         - Generate Python code that directly interacts with the `df` DataFrame to provide accurate results strictly based on the data in the dataset.
#                         - Do not make any assumptions or provide any example outputs.
#                         - Ensure all answers are derived from actual calculations on the `df` DataFrame.
#                         - Include concise comments explaining key steps in the code.
#                         - Exclude any visualization, plotting, or assumptions about the data.
#
#                         Example:
#
#                         Query: "How many rows have 'Column1' > 100?"
#                         Response:
#                         ```python
#                         # Count rows where 'Column1' > 100
#                         count_rows = df[df['Column1'] > 100].shape[0]
#
#                         # Output the result
#                         print(count_rows)
#                         ```
#
#                     2. Invalid or Non-Data Queries:
#                         If the user's query is unrelated to data processing or analysis, or it cannot be answered using the dataset, respond with an appropriate print statement indicating the limitation. For example:
#
#                         Query: "What is AI?"
#                         Response:
#                         ```python
#                         print("This question is unrelated to the uploaded data. Please ask a data-specific query.")
#                         ```
#
#                     3. Theoretical Concepts:
#                         If the user asks about theoretical concepts in data science or preprocessing (e.g., normalization, standardization), respond with a concise explanation. Keep the response focused and accurate.
#
#                         Example:
#
#                         Query: "What is normalization in data preprocessing?"
#                         Response:
#                         ```python
#                         print("Normalization is a data preprocessing technique used to scale numeric data within a specific range, typically [0, 1], to ensure all features contribute equally to the model.")
#                         ```
#
#                     User query: {query}.
#                 """
# #             )
# #
#             # Generate text-related code
#             code = generate_code(prompt_eng)
#             print("Generated code from AI (Text):")
#             print(code)
#
#             # Execute the generated code with the dataset
#             result = execute_py_code(code, df)
#             return JsonResponse({"answer": result}, status=200)
#
#     return HttpResponse("Invalid request method", status=405)


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
        file = request.FILES.get('file')
        # Handle user greetings with LLM response
        greetings = {"hi", "hello", "hey", "greetings"}
        if query in greetings:
            greeting_prompt = "Respond to the user greeting in a friendly and engaging manner."
            greeting_response = generate_code(greeting_prompt)
            return JsonResponse({"answer": markdown_to_html(greeting_response)})
        elif query and file:
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
            The insights should be categorized appropriately based on the nature of the comments.
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
        else:
            # Handle general topic-based queries with LLM response
            topic_prompt = f"Provide an informative response about '{query}' in a structured manner."
            topic_response = generate_code(topic_prompt)
            return JsonResponse({"answer": markdown_to_html(topic_response)})

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


# Sla breach code
import datetime
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings


@csrf_exempt
def process_query(request):
    if request.method == 'POST':
        try:
            # Check if a file is uploaded
            if 'file' not in request.FILES:
                return JsonResponse({"error": "No file uploaded."}, status=400)

            file = request.FILES['file']

            # Check if the file is a CSV
            if not file.name.endswith('.csv'):
                return JsonResponse({"error": "Invalid file format. Please upload a CSV file."}, status=400)

            # Read the CSV file directly from memory
            csv_data = pd.read_csv(file)

            # Process the data
            sorted_data = preprocess_data(csv_data)
            print(len(sorted_data))
            csv_data = calculate_time_differences(sorted_data)
            csv_data = calculate_sla_breach(csv_data)
            report_data = generate_report(csv_data)

            # Save the report to the media folder
            report_file_path = save_report_to_media(report_data)

            # Return the report data as JSON
            return JsonResponse({
                "message": "Report generated successfully.",
                "report_path": report_file_path,
                "report_data": report_data.to_dict(orient='records')
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": f"Processing failed: {str(e)}"}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method."}, status=405)


import pandas as pd
import datetime
import numpy as np

def preprocess_data(csv_data):
    # Convert 'Historical Status - Change Date' to datetime
    csv_data['Historical Status - Change Date'] = pd.to_datetime(
        csv_data['Historical Status - Change Date'], dayfirst=True, errors='coerce'
    )

    # Convert 'Historical Status - Change Time' to a 6-digit string and then to time
    csv_data['Historical Status - Change Time'] = csv_data['Historical Status - Change Time'].astype(str).str.zfill(6)
    csv_data['Historical Status - Change Time'] = pd.to_datetime(
        csv_data['Historical Status - Change Time'], format='%H%M%S', errors='coerce'
    ).dt.time

    # Combine date and time into a single datetime column
    csv_data['Change Datetime'] = csv_data.apply(
        lambda row: datetime.datetime.combine(row['Historical Status - Change Date'],
                                              row['Historical Status - Change Time'])
        if pd.notnull(row['Historical Status - Change Date']) and pd.notnull(row['Historical Status - Change Time'])
        else pd.NaT,
        axis=1
    )


    # Group the data by the 4th column (index 3) and sort each group by datetime
    grouped_data = csv_data.groupby(csv_data['Request - ID'])

    sorted_groups = []
    for _, group in grouped_data:
        group = group.sort_values(by=['Historical Status - Change Date', 'Change Datetime'])
        sorted_groups.append(group)

    # Concatenate the sorted groups back into a single DataFrame
    sorted_data = pd.concat(sorted_groups)
    print(sorted_data[['Request - ID', 'Historical Status - Change Date', 'Change Datetime']].head(20))

    return sorted_data

#New logic
import pandas as pd
import datetime


def calculate_working_hours(start, end):
    working_hours_start = datetime.time(14, 0, 0)  # 2 PM
    working_hours_end = datetime.time(23, 0, 0)  # 11 PM

    if start >= end:
        return 0.0

    total_hours = 0.0
    current_date = start.date()
    end_date = end.date()

    while current_date <= end_date:
        if current_date.weekday() >= 5:  # Skip weekends (Saturday=5, Sunday=6)
            current_date += datetime.timedelta(days=1)
            continue

        day_start = datetime.datetime.combine(current_date, working_hours_start)
        day_end = datetime.datetime.combine(current_date, working_hours_end)

        # Adjust interval_start and interval_end within the current day's working hours
        interval_start = max(start, day_start)
        interval_end = min(end, day_end)

        if interval_start < interval_end:
            delta = interval_end - interval_start
            total_hours += delta.total_seconds() / 3600

        current_date += datetime.timedelta(days=1)

    return total_hours


def calculate_time_differences(csv_data):
    # Initialize 'Change' column with 0.0 (float type)
    csv_data['Change'] = 0.0

    # Ensure 'Change Datetime' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(csv_data['Change Datetime']):
        csv_data['Change Datetime'] = pd.to_datetime(csv_data['Change Datetime'])

    # Group by 'Request - ID' and process each group
    for request_id, group in csv_data.groupby('Request - ID'):
        # Sort the group by 'Change Datetime'
        sorted_group = group.sort_values('Change Datetime')
        previous_time = None

        for index, row in sorted_group.iterrows():
            current_time = row['Change Datetime']

            if pd.isnull(current_time):
                csv_data.at[index, 'Change'] = 0.0
                continue

            is_weekend = current_time.weekday() >= 5

            if previous_time is None:
                # Handle first record in the group
                if is_weekend:
                    csv_data.at[index, 'Change'] = 0.0
                else:
                    # Check if current time is within working hours
                    if current_time.time() >= datetime.time(14, 0, 0):
                        start_of_day = current_time.replace(hour=14, minute=0, second=0, microsecond=0)
                        work_hours = calculate_working_hours(start_of_day, current_time)
                    else:
                        work_hours = 0.0
                    csv_data.at[index, 'Change'] = work_hours
                previous_time = current_time
            else:
                # Handle subsequent records
                work_hours = calculate_working_hours(previous_time, current_time)
                csv_data.at[index, 'Change'] = work_hours
                previous_time = current_time

    # Convert to total minutes as integer if needed
    # csv_data['Change'] = (csv_data['Change'] * 60).astype(int)
    filtered_data = csv_data[csv_data['Request - ID'] == 'A3033017L']

    # Select the required columns
    required_columns = [
        'Request - ID',
        'Historical Status - Change Date',
        'Change Datetime',
        'Change',
        'Historical Status - Status From',
        'Historical Status - Status To'
    ]

    # Print the first 20 rows of the filtered data
    print(filtered_data[required_columns])


    allowed_transitions = {
        ("Forwarded", "Assigned"),
        ("Forwarded", "Work in progress"),
        ("Assigned", "Work in progress"),
        ("Work in progress", "Suspended"),
        ("Work in progress", "Solved"),
        ("Suspended", "Solved"),
        ("Forwarded", "Suspended")
    }

    # Filter records based on allowed transitions
    csv_data= csv_data[
        csv_data[['Historical Status - Status From', 'Historical Status - Status To']]
        .apply(tuple, axis=1).isin(allowed_transitions)
    ]

    # Print debug information
    print("at_calculate_time_difference")
    print(csv_data.shape)
    # print(csv_data[['Request - ID', 'Historical Status - Change Date', 'Change Datetime', 'Change','Historical Status - Status From','Historical Status - Status To']].head(20))
    print("-------------------------------------------------------")
    filtered_data = csv_data[csv_data['Request - ID'] == 'A3033017L']

    # Select the required columns
    required_columns = [
        'Request - ID',
        'Historical Status - Change Date',
        'Change Datetime',
        'Change',
        'Historical Status - Status From',
        'Historical Status - Status To'
    ]

    # Print the first 20 rows of the filtered data
    print(filtered_data[required_columns])
    return csv_data


def calculate_sla_breach(csv_data):
    sla_mapping = {"P1 - Critical": 4, "P2 - High": 8, "P3 - Normal": 45, "P4 - Low": 90}
    csv_data['SLA Hours'] = csv_data['Request - Priority Description'].map(sla_mapping)
    csv_data['Total Elapsed Time'] = csv_data.groupby('Request - ID')['Change'].transform('sum').astype(int)
    csv_data['Time_to_breach'] = csv_data['SLA Hours'] - csv_data['Total Elapsed Time'].astype(int)

    # Set Time_to_breach to zero if Breached is "Yes"
    csv_data['Time_to_breach'] = np.where(
        csv_data['Total Elapsed Time'] > csv_data['SLA Hours'],
        0,
        csv_data['Time_to_breach']
    )

    csv_data['Breached'] = np.where(csv_data['Total Elapsed Time'] > csv_data['SLA Hours'], 'Yes', 'No')
    csv_data['Final_Status'] = csv_data.groupby('Request - ID')['Historical Status - Status To'].transform('last')
    print("at_calculate_sla_breach")
    print(csv_data.head(20))
    print(csv_data.shape)

    return csv_data


import pandas as pd


def parse_date_time(date_str, time_str):
    """Parses and combines date and time strings into a datetime object."""
    return pd.to_datetime(f"{date_str} {time_str}", errors='coerce')

import itertools
import pandas as pd
from datetime import datetime


def generate_report(csv_data):
    ticket_groups = {}

    for _, row in csv_data.iterrows():
        ticket_id = row['Request - ID']
        status_from = row['Historical Status - Status From']
        status_to = row['Historical Status - Status To']
        date = parse_date_time(row['Historical Status - Change Date'], row['Historical Status - Change Time'])

        if ticket_id not in ticket_groups:
            ticket_groups[ticket_id] = []

        ticket_groups[ticket_id].append({
            'row': row,
            'date': date,
            'statusFrom': status_from,
            'statusTo': status_to
        })

    print(len(ticket_groups))

    # Extract first record from each ticket group
    filtered_records = []
    for records in ticket_groups.values():
        if records:
            filtered_records.append(records[-1]['row'])  # Append only the last record from each group

    # Convert to DataFrame
    filtered_data = pd.DataFrame(filtered_records)
    print(filtered_data.head(10))


    # Prepare final report data
    report_data = filtered_data[[
        'Request - ID', 'Request - Priority Description', 'Request - Resource Assigned To - Name',
        'SLA Hours', 'Total Elapsed Time', 'Time_to_breach', 'Final_Status', 'Breached'
    ]].drop_duplicates()

    report_data.rename(columns={
        'Request - ID': 'Ticket',
        'Request - Priority Description': 'Priority',
        'Request - Resource Assigned To - Name': 'Assigned To',
        'SLA Hours': 'Allowed Duration(in Hours)',
        'Total Elapsed Time': 'Total Elapsed Time(in Hours)',
        'Time_to_breach': 'Time to Breach(in Hours)',
        'Final_Status': 'Status',
        'Breached': 'Breached'
    }, inplace=True)

    print("Final unique records:", report_data.shape)
    return report_data


def save_report_to_media(report_data):
    # Save the report to a fixed file name in the 'media' folder
    report_file_path = os.path.join(settings.MEDIA_ROOT, 'final_report.csv')
    report_data.to_csv(report_file_path, index=False)
    return report_file_path


import pandas as pd
import datetime
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
import io
import sys
import markdown
from bs4 import BeautifulSoup
from openai import OpenAI


@csrf_exempt
def sla_query(request):
    if request.method == 'POST':
        try:
            # Get the query from the form data
            query = request.POST.get("query")
            greetings = {"hi", "hello", "hey", "greetings"}

            if query in greetings:
                greeting_prompt = "Respond to the user greeting in a friendly and engaging manner."
                greeting_response = generate_coding_hi(greeting_prompt)
                return JsonResponse({"answer": markdown_to_html(greeting_response)})
            else:
                csv_file_path = os.path.join(settings.MEDIA_ROOT, 'final_report.csv')

                # Check if the file exists
                if not os.path.exists(csv_file_path):
                    return JsonResponse({"error": "CSV file not found. Please upload the file first."}, status=400)

                df = pd.read_csv(csv_file_path)

                # Generate CSV metadata
                csv_metadata = {"columns": df.columns.tolist()}
                metadata_str = ", ".join(csv_metadata["columns"])
                print(metadata_str)

                prompt_eng = (
                    f"""
                        You are a Python expert focused on answering user queries about data preprocessing. Always strictly adhere to the following rules:               

                        1. Data-Related Queries:
                            If the query is about data processing, assume the file {df.head(3)} is the data source and contains the following columns: {metadata_str}.

                            Strictly work with the data as it is in the CSV file. Do not perform any implicit conversions (e.g., hours to minutes) unless explicitly requested by the user query.
                            Example:
                            # Count tickets where 'Time to Breach' is less than or equal to 10 hours
                            breached_tickets_count = data[data['Time to Breach'] <= 10].shape[0]
                            If the query given to you is somewhat meaningless also,,try to analyze the important content in the query and generate the response based on the query.
                            For these queries, respond with Python code only, no additional explanations.
                            
                            If the user asks about the breached tickets don't consider the 'Status' column as breached, always consider the 'Breached' column as Yes.
                            
                            The code should:
                            Load {csv_file_path} using pandas.
                            Perform operations to directly address the query.
                            Exclude plotting, visualization, or other unnecessary steps.
                            Include comments for key steps in the code.
                            Example:

                            Query: "How can I filter rows where 'Column1' > 100?"
                            Response:
                            python
                            Copy code
                            import pandas as pd

                            # Load the dataset
                            data = pd.read_csv('{csv_file_path}')

                            # Filter rows where 'Column1' > 100
                            filtered_data = data[data['Column1'] > 100]

                            # Output the result
                            print(filtered_data)

                        When returning data retrieved from the database, always aim to present it in a **tabular format** for clarity and better readability, if applicable.Generate the response in HTML table format. Use proper <table>, <thead>, <tbody>, <tr>, and <td> tags. Ensure the table structure is well-formed and include the following columns which will be compatable to React.
                        If request is asked for tickets or breached tickets or list of tickets , please only mention Ticket,Priority,Assigned To,Allowed Duration,Total Elapsed Time,Time to Breach,Status,Breached in the response within the tabular format.

                        Never reply with: "Understood!" or similar confirmations. Always directly respond to the query following the above rules.

                        User query is {query}.
                    """
                )
                print("Prompt from AI:", prompt_eng)
                code = generate_code1(prompt_eng)
                print("Generated code from AI (Text):")
                print(code)

                # Execute the generated code with the dataset
                result = execute_py_code(code, df)
                return JsonResponse({"answer": result})

        except Exception as e:
            return JsonResponse({"error": f"Processing failed: {str(e)}"}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method."}, status=405)


def generate_code1(prompt_eng):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_eng}
        ]
    )
    all_text = ""
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    return code


def execute_py_code(code, df):
    # Create a string buffer to capture the output
    buffer = io.StringIO()
    sys.stdout = buffer

    # Create a local namespace for execution
    local_vars = {'df': df}

    try:
        # Execute the code
        exec(code, globals(), local_vars)

        # Get the captured output
        output = buffer.getvalue().strip()

        # If there's no output, try to get the last evaluated expression
        if not output:
            last_line = code.strip().split('\n')[-1]
            if not last_line.startswith(('print', 'return')):
                output = eval(last_line, globals(), local_vars)
                output = str(output)
    except Exception as e:
        output = f"Error executing code: {str(e)}"
    finally:
        # Reset stdout
        sys.stdout = sys.__stdout__

    return output


def generate_coding_hi(prompt_eng):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing actionable insights."},
            {"role": "user", "content": prompt_eng}
        ]
    )
    return response.choices[0].message.content.strip()


def markdown_to_html1(md_text):
    # Convert Markdown to HTML
    html_content = markdown.markdown(md_text)

    # Parse the HTML to modify headings
    soup = BeautifulSoup(html_content, "html.parser")

    # Apply blue color to all heading tags
    for tag in soup.find_all(['p']):
        tag['style'] = "color: blue;"

    return str(soup)
