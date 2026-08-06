import sqlite3
import os

p = 'experiments/experiments.sqlite3'
print('exists', os.path.exists(p))
con = sqlite3.connect(p)
print('tables', con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
print('groups', con.execute("SELECT name FROM experiment_groups").fetchall())
print('runs', con.execute("SELECT name, group_id FROM runs LIMIT 10").fetchall())
