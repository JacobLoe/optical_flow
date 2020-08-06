class IdMapper(object):
    pass


class TSVIdMapper(IdMapper):

   def __init__(self, tsv_file):
      with open(tsv_file, 'r') as in_tsv:
         self._ids = dict([(l.split('\t')[0].strip().replace('"',''), l.split('\t')[1].strip().replace('"','')) for l in in_tsv.readlines()[1:]])

   def get_filename(self, videoid):
      return self._ids[videoid]

   def get_ids(self):
      return list(self._ids.keys())
