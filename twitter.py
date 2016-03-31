from TwitterSearch import *
import datetime, dateutil.parser, json

LATLONG = (32.11766, 20.12928) # Benghazi

def main():
    tso = TwitterSearchOrder()
    tso.set_keywords([''])
    tso.set_geocode(LATLONG[0], LATLONG[1], 50)
    tso.set_include_entities(False)

    ts = TwitterSearch(
        consumer_key='N80C0jvGBfPnmIyvrdWAVMFad',
        consumer_secret='TIfAaF2Fx9slaQVg19gHX4aNM5BmuhtqvbTxsRvdckcAbSeqgA',
        access_token='374264753-1Tlw3ovzBmPbzmq6ttA83csCNLNOogfOuZUJA1tk',
        access_token_secret='1b53oL9YRj1M2LYswIkwfAFQDBAfyfATzV35j7XZx0u0H'
    )

    tweets = []
    today = datetime.date.today()
    for tweet in ts.search_tweets_iterable(tso):
        if dateutil.parser.parse(tweet['created_at']).date() < today: continue
        tweets.append(tweet)

    with open('log/{:%Y-%m-%d}.log'.format(today), 'wb') as log:
        log.write(json.dumps(tweets))

if __name__ == "__main__":
    main()
