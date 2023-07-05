import pandas as pd
import os
import openai

openai.api_key = "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"


def GPT_analysis(sentence):
  response = openai.Completion.create(
  model="text-ada-001",
  prompt=f"Decide whether a newsheadline's sentiment is positive, neutral, or negative.\n\nnewsheadline: \"{sentence}\"\nSentiment:",
  temperature=0,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0.5,
  presence_penalty=0
)
  output = response['choices'][0]['text']
  print(output)
  return output
  


SPInput = "Output/NASDAQ"
SPOutput = "Output/NASDAQ_GPT" 
if not os.path.exists(SPOutput):
    os.makedirs(SPOutput)
    
    
NASDAQinput = "Output/S&P"
NASDAQoutput = "Output/S&P_GPT"
if not os.path.exists(NASDAQoutput):
    os.makedirs(NASDAQoutput)
    

# Folder path containing the CSV files
marketInput = "Output/market"
marketOutput = "Output/marketGPT"
if not os.path.exists(marketOutput):
    os.makedirs(marketOutput)

# Iterate through the files in the folder



def GPT_analysis_for_folder(inputpath, outputpath): 
    print(os.listdir(inputpath))
    for filename in os.listdir(inputpath):
    
        if filename.endswith(".csv"):
            file_path = os.path.join(inputpath, filename)
            print(f'start processing {file_path}')
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get the column names that end with "_Title" or "_Subtitle"
            relevant_columns = [col for col in df.columns if col.endswith(("_Title", "_Subtitle"))]
            print(f'relevant_columns are {relevant_columns}')
            
            # Apply GPT_process() to relevant columns
            for column in relevant_columns:
                new_column_name = column + "_GPT"
                df[new_column_name] = df[column].apply(GPT_analysis)
            
            # Save the updated DataFrame to a new file
            df.to_csv(f'{outputpath}/{filename}.csv', index=False)
            print(f'{filename} saved')
    print(f'{inputpath} finished')
            
            
GPT_analysis_for_folder(NASDAQinput, NASDAQoutput)
GPT_analysis_for_folder(SPInput, SPOutput)
GPT_analysis_for_folder(marketInput, marketOutput)

  

  