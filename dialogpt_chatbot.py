from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# DialoGPT model
model_name = "microsoft/DialoGPT-large"
# model_name = "microsoft/DialoGPT-medium"
# model_name = "microsoft/DialoGPT-small"
# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# load the model
model = AutoModelForCausalLM.from_pretrained(model_name)


def get_chatbot_response(input_text, chat_history_ids=None, step=0):
    # take user input
    text = input_text
    # encode the input and add end of string token
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    # concatenate new user input with chat history (if there is)
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
    # generate a bot response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1500,
        do_sample=True,
        top_k=100,
        # top_p=0.92,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    # decode the bot response
    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return output, chat_history_ids


if __name__ == "__main__":
    # number of times you want to talk with the bot
    n_steps = 5
    chat_history_ids = None
    for step in range(n_steps):
        # take the text from the input
        text = input(">> ")
        # retrieve the bot response and print it
        response, chat_history_ids = get_chatbot_response(text, chat_history_ids, step)
        # print the chatbot response
        print("DialoGPT:", response)