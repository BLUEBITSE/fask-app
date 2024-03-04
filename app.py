from flask import Flask, request, jsonify
from mcq_generator import MCQGenerator

app = Flask(__name__)
mcq_generator = MCQGenerator()

@app.route('/api/generate-mcq', methods=["POST"])
def generate_mcq_api():
    if request.method == 'POST':
        # Get text data from the request
        text_data = request.form['text']

        if text_data:
            # Generate MCQs from the text data
            mcqs = mcq_generator.generate_mcqs(text_data)

            # Format the generated MCQs
            formatted_mcqs = []
            for mcq in mcqs:
                formatted_mcq = {
                    "question": mcq['question'],
                    "correct_answer": mcq['answer'],
                    "options": mcq['options']
                }
                formatted_mcqs.append(formatted_mcq)

            # Return the formatted MCQs as JSON response
            return jsonify({"mcqs": formatted_mcqs})
        else:
            return jsonify({"error": "Text parameter is missing"}), 400
    else:
        return jsonify({"error": "Only POST requests are allowed"}), 405


