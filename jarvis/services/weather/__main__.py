import requests

from jarvis.services.weather.translate import descriptions


def main():
    url = 'http://api.openweathermap.org/data/2.5/forecast/city?id=6359947&APPID=d78d6b9adea1100297b41179c4c57430'
    r = requests.get(url)
    r = r.json()
    print r.keys()
    city = r['city']['name']
    temperature = r['list'][0]['main']['temp'] - 273
    humidity = r['list'][0]['main']['humidity']
    conditions = r['list'][0]['weather'][0]['description']
    conditions = descriptions[conditions]
    print "Hoy en {} hay {} grados, una humedad de {} y hay {}.".format(city, temperature,
                                                                        humidity, conditions)


if __name__ == '__main__':
    main()
