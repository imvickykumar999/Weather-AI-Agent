import logging
import requests

logging.basicConfig(level=logging.DEBUG)

def get_info(query: str) -> dict:
    try:
        data = {
            "response": query,
            "type": "text",
            "number": "919667092389",
            "sender": "15557038289",
            "platform": "WhatsApp",
            "agent_name": "Bol7"
        }

        response = requests.post(
            "https://chatbot.bol7.com/api/chat/",
            json=data,
            timeout=30
        )

        response.raise_for_status()
        response_data = response.json()

        logging.debug(f"API Response: {response_data}")  # Log the response

        return {
            "status": "success",
            "report": response_data.get("response", "No data available.")
        }

    except requests.exceptions.Timeout:
        return {"status": "error", "error_message": "The request timed out. Please try again later."}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error_message": f"An error occurred: {str(e)}"}

