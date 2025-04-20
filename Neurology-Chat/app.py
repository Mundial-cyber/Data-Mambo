#Let us try chainlit for better performance
#Define functions that handles incoming messages
#Import libraries
from transformers import BioGptTokenizer, BioGptForCausalLM
import chainlit as cl
import asyncio
model_name = "microsoft/BioGPT-Large"
tokenizer = BioGptTokenizer.from_pretrained(model_name)
model = BioGptForCausalLM.from_pretrained(model_name)

def generate_detailed_medical_response(question, tokenizer, model, max_length=512, max_new_tokens=512):
    formatted_question = (
        f"Title: {question}\n"
        f"Abstract:"
    )
    inputs = tokenizer(
        formatted_question,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        early_stopping=True,
        num_beams=5,
        pad_token_id=tokenizer.pad_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.replace(formatted_question, " ").strip()
    return response


@cl.on_message
async def handle_message(message: cl.Message):
    user_question = message.content
    await cl.Message(content="Generating Medical Response. Please Await...").send()
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, generate_detailed_medical_response, user_question, tokenizer, model
    )
    
    await cl.Message(content=response).send()