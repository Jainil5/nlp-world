from flask import Flask
from flask_restful import Api, Resource, reqparse
from sentiment_cardiff import get_sentiment

app = Flask(__name__)
api = Api(app)

class Sentiment(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('input', type=str, required=True)
        args = parser.parse_args()
        input = args['input']
        response = str(get_sentiment(input))
        return {"response": response}

api.add_resource(Request, "/sentiment")


class Request(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('input', type=str, required=True)
        args = parser.parse_args()
        input = args['input']
        response = str(get_sentiment(input))
        return {"response": response}

api.add_resource(Request, "/sentiment")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
