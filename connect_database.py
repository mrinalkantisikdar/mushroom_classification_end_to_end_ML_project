from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

cloud_config= {
  'secure_connect_bundle': os.path.join(os.getcwd(), 'secure-connect-mushrooms.zip')
}
auth_provider = PlainTextAuthProvider('OHzAlMtmSMGeBtdPqdZcrUHl', 'yy_PfrJIa-49jCfM-sRCb+d7xcBZdbZ8fJ6h0q.+n.b_8IFgfbtvsvNec0pK,FTqLt9NBXYg-iPPqh5JlEmXLlrb_ZFK1DosAMD8yFOE7G9e_PQYO4,1eI5B4BzZPnnG')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session= cluster.connect('ineuron_mlprojects')

data = session.execute("SELECT * FROM mushroom_csv;")

# data = pd.DataFrame([d for d in data])


'''
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

auth_provider = PlainTextAuthProvider(username=CASSANDRA_USER, password=CASSANDRA_PASS)
cluster = Cluster(contact_points=[CASSANDRA_HOST], port=CASSANDRA_PORT,
        auth_provider=auth_provider)

session = cluster.connect(CASSANDRA_DB)
data = session.execute("SELECT * FROM <table_name>;")

df = pd.DataFrame([d for d in data])
'''  

'''
session= cluster.connect('ineuron_mlprojects')

data = session.execute("SELECT * FROM mushroom_csv;")

df = pd.DataFrame([d for d in data])

'''