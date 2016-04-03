import numpy as np
import sqlite3
from datetime import date, timedelta, datetime

ACLED_KEYS = ['GWNO','EVENT_ID_CNTY','EVENT_ID_NO_CNTY','EVENT_DATE','YEAR','TIME_PRECISION','EVENT_TYPE','ACTOR1','ALLY_ACTOR_1','INTER1','ACTOR2','ALLY_ACTOR_2','INTER2','INTERACTION','COUNTRY','ADMIN1','ADMIN2','ADMIN3','LOCATION','LATITUDE','LONGITUDE','GEO_PRECISION','SOURCE','NOTES','FATALITIES']

COUNTRY = 'egypt'
WINDOW_SIZE = 14 # days
NUM_CLUSTERS = 12
CROSSVAL_K = 10

def cluster_latlong():
    (conn, c) = get_db()
    from scipy.cluster.vq import kmeans, vq
    from itertools import cycle
    from pylab import plot, show

    print 'Retrieving city coordinates...'
    latlong = []
    for fields in c.execute('SELECT latitude, longitude FROM events').fetchall():
        latlong.append(fields)
    latlong = np.array(latlong)

    print 'Clustering...'
    clusters, _ = kmeans(latlong, NUM_CLUSTERS)
    idx, _ = vq(latlong, clusters)
    np.save('clusters', clusters)
    np.save('cluster_mapping', idx)

    # print 'Plotting...'
    # colors = cycle('bryckmwbryckmw')
    # for i in range(num_clusters):
    #     plot(latlong[idx==i,0], latlong[idx==i,1], 'o' + colors.next())
    # plot(clusters[:,0],clusters[:,1],'sg',markersize=8)
    # show()


def get_db():
    conn = sqlite3.connect('{}.db'.format(COUNTRY))
    c = conn.cursor()
    return (conn, c)


def csv_to_sqlite():
    import csv

    db = []
    print 'Loading CSV...'
    with open('{}.csv'.format(COUNTRY), 'rU') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            try:
                row = {ACLED_KEYS[i]: row[i] for i in range(len(ACLED_KEYS))}
                row['LATITUDE'] = float(row['LATITUDE'])
                row['LONGITUDE'] = float(row['LONGITUDE'])
                row['EVENT_DATE'] = datetime.strptime(row['EVENT_DATE'], '%d/%m/%Y')
                db.append(row)
            except Exception:
                pass

    rows = [[i, r['EVENT_DATE'], r['LATITUDE'], r['LONGITUDE']]
            for (i, r) in enumerate(db)]

    (conn, c) = get_db()
    c.execute('CREATE TABLE events (id int, event_date datetime, latitude real, longitude real)')
    c.executemany('INSERT INTO events VALUES(?,?,?,?)', rows)
    conn.commit()
    conn.close()


def daterange(start_date, end_date):
    """ Returns an iterator for all days between [start_date] and [end_date]. """
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


def train_classifier():
    (conn, c) = get_db()
    clusters = np.load('clusters.npy')
    indices = np.load('cluster_mapping.npy')

    print 'Generating samples...'
    xs = []
    ys = []
    for day in daterange(date(2013, 1, 1), date(2015, 12, 31)):
        c.execute('SELECT id, event_date FROM events WHERE event_date >= ? '
                  'AND event_date <= ?',
                  [day - timedelta(days=WINDOW_SIZE), day + timedelta(days=1)])
        x = [0 for _ in range(len(clusters))]
        y = [0 for _ in range(len(clusters))]
        for (idx, event_date_s) in c.fetchall():
            cluster = indices[idx]
            event_date = datetime.strptime(event_date_s, '%Y-%m-%d 00:00:00')
            if event_date.date() == day:
                y[cluster] = 1
            else:
                x[cluster] += 1 #WINDOW_SIZE - (event_date.date() - day).days
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)

    print 'Training...'
    from sklearn import cross_validation as cv
    from sklearn.naive_bayes import GaussianNB
    from sklearn.base import clone
    kf = cv.KFold(len(ys), CROSSVAL_K)

    # cluster_error = 0.0
    # gnb = GaussianNB()
    # for cy in range(len(clusters)):
    #     ys2 = ys[:, cy]
    #     scores = cv.cross_val_score(gnb, xs, ys2, cv=kf)
    #     print 'k-CV for {}: {} (+/- {:2})'.format(cy, scores.mean(), scores.std() * 2)
    #     cluster_error += scores.mean()
    # print 'Average k-CV: {}'.format(cluster_error / len(clusters))

    from sknn import mlp
    clf = mlp.Classifier(
        layers=[mlp.Layer("Sigmoid", units=NUM_CLUSTERS),
                mlp.Layer("Softmax")],
        learning_rate=0.02,
        n_iter=10)
    clf.fit(xs, ys)
    for i in range(1000):
        print clf.predict(np.array([xs[i,:]]))
    # for train, test in kf:
    #     print 'Train xs', xs[train]
    #     print 'Train ys', ys[train]
    #     c = clone(clf)
    #     c.fit(xs[train], ys[train])
    #     print 'Predicted', c.predict(xs[test])
    #     print 'Actual', ys[test]
    #     #print (c.predict(xs[test]) != ys[test]).mean()


def main():
    #cluster_latlong()
    train_classifier()


if __name__ == "__main__":
    main()
