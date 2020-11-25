from flask import Flask, request
from flask_restful import Resource, Api
import requests, json
import prediction_model
from datetime import date

app = Flask(__name__)
api = Api(app)

cntryCode = {'AUSTRALIA':'AUD', 'EURO AREA':'EUR', 'NEW ZEALAND':'NZD', 'UNITED KINGDOM':'GBP', 'BRAZIL':'BRL', 'CANADA':'CAD', 'CHINA':'CNY', 'HONG KONG':'HKD', 'INDIA':'INR', 'KOREA':'KRW', 'MEXICO':'MXN', 'SOUTH AFRICA':'ZAR', 'SINGAPORE':'SGD', 'DENMARK':'DKK', 'JAPAN':'JPY', 'MALAYSIA':'MYR', 'NORWAY':'ISK', 'SWEDEN':'SEK', 'SWITZERLAND':'CHF', 'THAILAND':'THB'}

rates = {'date': date.today().strftime('%Y-%m-%d')}

class Predict(Resource):
	def get(self):
		return json.loads(json.dumps(rates))

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
	cntryNames = ['AUSTRALIA', 'EURO AREA', 'NEW ZEALAND']
	# cntryNames = ['AUSTRALIA', 'EURO AREA', 'NEW ZEALAND', 'UNITED KINGDOM', 'BRAZIL', 'CANADA', 'CHINA', 'HONG KONG', 'INDIA', 'KOREA', 'MEXICO', 'SOUTH AFRICA', 'SINGAPORE', 'DENMARK', 'JAPAN', 'MALAYSIA', 'NORWAY', 'SWEDEN', 'SWITZERLAND', 'THAILAND']
	for country in cntryNames:
		rates[cntryCode[country]] = prediction_model.returnRates(country)
    
	app.run(debug=True)