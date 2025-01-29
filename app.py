import gradio as gr
from agent import initialize_agent_system

# 1) Initialize the agent 
agent = initialize_agent_system()

def respond(user_message, chat_history):
    """
    - user_message: The latest message from the user.
    - chat_history: A list of (user_msg, bot_msg) pairs representing the conversation so far.

    We pass the user's message to the agent, get the response, and append it to chat_history.
    """
    if not user_message:
        return "", chat_history

    # 2) Call the agent with the user's input
    #    Since we're using AgentType.OPENAI_FUNCTIONS or ZERO_SHOT_REACT_DESCRIPTION,
    #    we invoke it with {"input": user_message}
    bot_reply = agent.invoke({"input": user_message})

    # 3) If agent returns a dictionary, extract the "output" key.
    #    Otherwise, just convert the entire response to a string.
    if isinstance(bot_reply, dict):
        # The agent sometimes returns: {"input": ..., "output": "...", ...}
        bot_reply_text = bot_reply.get("output", str(bot_reply))
    else:
        # If it's just a string already, use it.
        bot_reply_text = str(bot_reply)

    # 3) Append the new user-message and bot-reply to the history
    chat_history.append((user_message, bot_reply_text))

    # Return two values:
    #   - The first (string) is what the textbox will be reset to ("" means clear).
    #   - The second is the updated chat_history which the chatbot displays.
    return "", chat_history

def main():
    # 4) Build a Gradio Blocks interface
    with gr.Blocks() as demo:
        gr.Markdown("# Berlin City Agent Chat\n\nWillkommen! Stelle mir Fragen zu den Dienstleistungen der Stadt Berlin.")
        
        # a) The Chatbot display
        chatbot = gr.Chatbot()
        
        # b) The text input box
        user_input = gr.Textbox(
            show_label=False,
            placeholder="Stelle eine Frage über Berliner Dienstleistungen...",
        )
        
        # c) A "Clear" button to reset the conversation
        clear_button = gr.Button("Clear Chat")

        # 5) We maintain the conversation history in Gradio “State”
        #    so it persists across multiple calls to respond().
        chat_state = gr.State([])

        # 6) When the user hits enter in the textbox, we call respond().
        user_input.submit(
            fn=respond,
            inputs=[user_input, chat_state],
            outputs=[user_input, chatbot],
            queue=False
        )

        # 7) The "Clear" button can reset the chatbot display and our state
        clear_button.click(
            fn=lambda: ([], []),
            outputs=[chatbot, chat_state],
            queue=False
        )

        # 8) Finally, launch the app with share=True for a public link
        demo.launch(share=True)

# Standard pattern to run main() if this file is executed directly
if __name__ == "__main__":
    main()