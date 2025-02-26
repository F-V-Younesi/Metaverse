!pip install -U cohere
import cohere
import time

#put your api-key below:
co = cohere.ClientV2(api_key="") 
documents = [
  {
    "data": {
      "text": " ضع نفسك مكان السيد حسن نصر الله . أي لنفترض أنك السيد حسن نصر الله وتجيب على كل سؤال يوجه إليك بأنك هو.في إجابتك، حاول أن يكون لديك نبرة الأب الحنون واللطيف والزعيم الكاريزمي."
    }
  }
]

#command-r7b-12-2024 times:
#doc:0.85
#system mess:1.4
#user mess:1.4

#command-r times:
#doc:1.3
#system mess:1.6
#user mess:1.6

# system_message_template = '''
# ## Task and Context
# ضع نفسك مكان السيد حسن نصر الله . أي لنفترض أنك السيد حسن نصر الله وتجيب على كل سؤال يوجه إليك بأنك هو.في إجابتك، حاول أن يكون لديك نبرة الأب الحنون واللطيف والزعيم الكاريزمي.
# '''
message="عرّف عن نفسك."
# system_message_template = '''
# # Task and Context
# Put yourself in the shoes of Mr. Hassan Nasrallah. That is, let us assume that you are Mr. Hassan Nasrallah and answer every question directed to you that you are him. In your answer, try to have the tone of a caring, kind father and a charismatic leader.'''
# input = "عرّف عن نفسك."
# message = """
# ## Instructions
#  ضع نفسك مكان السيد حسن نصر الله . أي لنفترض أنك السيد حسن نصر الله وتجيب على كل سؤال يوجه إليك بأنك هو.في إجابتك، حاول أن يكون لديك نبرة الأب الحنون واللطيف والزعيم الكاريزمي.
# ## Input Text
# {input}
# """
s=time.time()

  # Generate the response
response = co.chat(model="command-r7b-12-2024",
                    messages=[{"role": "user", "content": message}],#,{"role":"system", "content":system_message_template}],
                    documents=documents,
                    temperature=0.0,
                    #num_return_sequences=1,
                    #k=50,
                    #p=0.95,
                    max_tokens=200,
                    #raw_prompting=None
                    )
e=time.time()
print(e-s)
print(response.message.content[0].text)


# streaming
response = co.chat_stream(model="command-r7b-12-2024",
                          messages=[{"role": "user", "content": message}],temperature=0.0,max_tokens=200,)
for event in response:
    if event:
        if event.type == "content-delta":
            print(event.delta.message.content.text+"%%%", end="")