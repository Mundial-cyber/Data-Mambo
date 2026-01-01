#Let us try chainlit for better performance
#Define functions that handles incoming messages
#Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import chainlit as cl
import asyncio
bio_model_name = "microsoft/BioGPT-Large"
bio_model_tokenizer = AutoTokenizer.from_pretrained(bio_model_name)
bio_model = AutoModelForCausalLM.from_pretrained(bio_model_name)

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

mistral_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_name)

#Simplify for a non-medical audience
def simplified_medical_answer(raw_answer, tokenizer, model, max_length=512, max_new_tokens=512):
    simplified_answer = f"Explain this for a general audience:\n\n{raw_answer}\n\nSimplified Answer"
    inputs = tokenizer(
        simplified_answer,
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
    simplified_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    simplified_response = simplified_decoded.replace(simplified_answer, " ").strip()
    return simplified_response


@cl.on_message
async def handle_message(message: cl.Message):
    user_question = message.content
    await cl.Message(content="Generating Medical Response. Please wait...").send()
    loop = asyncio.get_event_loop()
    #Generate a detailed medical response using BioGPT
    detailed_response = await loop.run_in_executor(
        None, generate_detailed_medical_response, user_question, bio_model_tokenizer, bio_model
    )
    simplified_response = await loop.run_in_executor(
        None, simplified_medical_answer, detailed_response, mistral_tokenizer, mistral_model
    )
    
    await cl.Message(content=simplified_response).send()