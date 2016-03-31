import numpy as np
import sqlite3
from datetime import date, timedelta, datetime

ACLED_KEYS = ['GWNO','EVENT_ID_CNTY','EVENT_ID_NO_CNTY','EVENT_DATE','YEAR','TIME_PRECISION','EVENT_TYPE','ACTOR1','ALLY_ACTOR_1','INTER1','ACTOR2','ALLY_ACTOR_2','INTER2','INTERACTION','COUNTRY','ADMIN1','ADMIN2','ADMIN3','LOCATION','LATITUDE','LONGITUDE','GEO_PRECISION','SOURCE','NOTES','FATALITIES']

COUNTRY = 'egypt'
WINDOW_SIZE = 30 # days
NUM_CLUSTERS = 12

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
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def kfold_cv(Classifier, xs, ys, k):
    pk = len(xs) / k
    avg_error = 0.0
    for i in range(k):
        cls = Classifier()
        ki = pk * i
        kj = pk * (i + 1)
        xs_train = xs[:ki] + xs[kj:]
        ys_train = ys[:ki] + ys[kj:]
        xs_test = xs[ki:kj]
        ys_test = ys[ki:kj]
        cls.fit(xs_train, ys_train)
        y_pred = cls.predict(xs_test)
        (wrong,) = (ys_test != y_pred).sum(),
        total = len(xs_test)
        error = float(wrong) / total
        avg_error += error
    return avg_error / k

def train_classifier():
    (conn, c) = get_db()
    clusters = np.load('clusters.npy')
    indices = np.load('cluster_mapping.npy')

    cluster_error = 0.0
    for cy in range(len(clusters)):
        xs = []
        ys = []
        for day in daterange(date(2013, 1, 1), date(2015, 12, 31)):
            c.execute('SELECT id, event_date FROM events WHERE event_date >= ? AND event_date <= ?',
                      [day - timedelta(days=WINDOW_SIZE), day + timedelta(days=1)])
            x = [0 for _ in range(len(clusters))]
            y = 0
            for (idx, event_date_s) in c.fetchall():
                cluster = indices[idx]
                event_date = datetime.strptime(event_date_s, '%Y-%m-%d 00:00:00')
                if event_date.date() == day:
                    if cluster == cy: y = 1
                else:
                    x[cluster] += 1 #WINDOW_SIZE - (event_date.date() - day).days
            xs.append(x)
            ys.append(y)

        from sklearn.naive_bayes import GaussianNB
        k = 10
        kerr = kfold_cv(GaussianNB, xs, ys, k)
        print 'k-CV for {}: {}'.format(cy, kerr)
        cluster_error += kerr
    print 'Average k-CV: {}'.format(cluster_error / len(clusters))

def main():
    #cluster_latlong()
    train_classifier()


if __name__ == "__main__":
    main()
