#In english Lang: best is c4ai-aya-expanse-32b in time

#In persian: best is c4ai-aya-expanse-8b in time (25% faster)

#best for rag: command-r , command-r-03-2024

import cohere
# #chat model:
co = cohere.ClientV2(api_key="dnsaqjNs9bwBvLtlPDiEGw1n1sWux6R7kTY7wcHA") 

#build and introduce Dataset to model:
chat_dataset = co.datasets.create(name="chat-dataset",
                                   data=open("path/to/train.jsonl", "rb"),
                                   type="chat-finetune-input")
print(co.wait(chat_dataset))
print(co.wait(chat_dataset).dataset.validation_status)
chat_dataset_with_eval = co.datasets.create(name="chat-dataset-with-eval",data=open("path/to/train.jsonl", "rb"), eval_data=open("path/to/eval.jsonl", "rb"),type="chat-finetune-input")
print(co.wait(chat_dataset_with_eval))

#Input RAG or give Documents to model:
documents = [
  {
    "data": {
      "text": "درصورتیکه که در مورد بازی بیلدر سوالی انجام شد، در بازی بیلدر، امتیاز 0 تا 50 پایین است، امتیاز 50 تا 70 متوسط است و امتیاز بالای 70 خوب است."
    }
  }
]


#model without streaming:
def llm(input):
    #@@@without stream:
    system_message_template = '''
    ## Task and Context
    You are a friendly assistant and guide and a companion to help people.
    '''
    message = f"""
      ## Instructions
      فرض کن تو راهنمای من هستی، به این سوال در حداکثر دو جمله پاسخ بده. درصورتیکه محتوای سوال با محتوای documents مرتبط بود، از محتوای documents برای پاسخ دادن استفاده کن. در غیر این صورت، از دانش خودت استفاده کن:
      ## Input Text
      {input}
      """
    response = co.chat(model="command-r",
                       messages=[{"role": "user", "content": message},
                                 {"role": "system", "content": system_message_template}],
                       documents=documents,
                       temperature=0.0,
                       # num_return_sequences=1,
                       # k=50,
                       # p=0.95,
                       max_tokens=200,
                       # raw_prompting=None
                       )
    return response.message.content[0].text


def emo_res(input,emotion):
  #@@@ respond base on emotion of user, without stream:
  system_message_template = '''
  ## Task and Context
  You are a friendly assistant and guide and a companion to help people.
  '''
  message = f"""
  ## Instructions
  فرض کن تو راهنمای من هستی، به این سوال در حداکثر دو جمله پاسخ بده. درصورتیکه محتوای سوال با محتوای documents مرتبط بود، از محتوای documents برای پاسخ دادن استفاده کن. در غیر این صورت، از دانش خودت استفاده کن:
  ## Input Text
  من {emotion} هستم. {input}
  """

  # Generate the response
  response = co.chat(model="command-r",
                    messages=[{"role": "user", "content": message},{"role":"system", "content":system_message_template}],
                    documents=documents,
                    temperature=0.0,
                    #num_return_sequences=1,
                    #k=50,
                    #p=0.95,
                    max_tokens=200,
                    #raw_prompting=None
                    )
  return response.message.content[0].text


# streaming
def llm(input):
  system_message_template = '''
    ## Task and Context
    You are a friendly assistant and guide and a companion to help people.
    '''
  message = f"""
      ## Instructions
      فرض کن تو راهنمای من هستی، به این سوال در حداکثر دو جمله پاسخ بده. درصورتیکه محتوای سوال با محتوای documents مرتبط بود، از محتوای documents برای پاسخ دادن استفاده کن. در غیر این صورت، از دانش خودت استفاده کن:
      ## Input Text
      {input}
      """
  response = co.chat_stream(model="command-r-plus-08-2024",
                            messages=[{"role": "user", "content": message},{"role": "system", "content": system_message_template}],documents=documents,
                            temperature=0.0,
                           # num_return_sequences=1,
                           # k=50,
                           # p=0.95,
                           max_tokens=200,
                           # raw_prompting=None
                           )
  for event in response:
      if event:
          if event.type == "content-delta":
              return event.delta.message.content.text
