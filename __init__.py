# CEIC R wrapper for python
# Author: Ronald Yip

import rpy2
from rpy2.robjects.conversion import Converter, localconverter
from rpy2.robjects import default_converter
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2 import robjects
from rpy2.robjects.packages import importr
import re
import pandas as pd
import numpy as np
import country_converter as coco
from warnings import warn
import datetime
import dateutil.relativedelta as relativedelta

ceic = importr('ceic')
base = importr('base')
zoo = importr('zoo')
stats = importr('stats')

#
# R-Python object converter
##

ts_converter = Converter('ts converter')
base_converter = default_converter + numpy2ri.converter + pandas2ri.converter

@ts_converter.rpy2py.register(rpy2.rinterface.FloatSexpVector)
@ts_converter.rpy2py.register(rpy2.rinterface.IntSexpVector)
def ts_convert(obj):
    if stats.is_ts(obj)[0]:
        return ts2series(obj)
    elif zoo.is_zoo(obj)[0]:
        return zoo2series(obj)
    else:
        return base_converter.rpy2py(obj)

'''
@ts_converter.py2rpy.register(pd.Series)
def series2ts(obj):
    if type(obj.index) == pd.RangeIndex:
        return robjects.r.ts(robjects.r.c(*obj.tolist()), start=obj.index.start, frequency=1/obj.index.step)
    return base_converter.py2rpy(obj)
'''

rcv = default_converter + numpy2ri.converter + pandas2ri.converter + ts_converter

def ts2series(ts, month_end=True, period_end=True, index='auto'):
    _raise('Wrong value of index param. ') if not index in ['auto', 'numeric', 'range', 'date'] else None
    tsp = [x for x in robjects.r.tsp(ts)]
    if index == 'range':
        return pd.Series(ts)
    if index == 'numeric':
        return pd.Series(ts, index=np.linspace(tsp[0], tsp[1], round((tsp[1] - tsp[0]) * tsp[2] + 1)))
    if index == 'auto' and index != 'date' and (1 / tsp[2]).is_integer() and tsp[0].is_integer():
        tsp[1] += 1 / tsp[2]
        return pd.Series(ts, index=range(int(tsp[0]), int(tsp[1]), int(1 / tsp[2])))
    if index in ('auto', 'date') and tsp[2] in (12.0, 4.0, 1.0) and (tsp[0] * tsp[2]).is_integer():
        tsp[1] += 1 / tsp[2]
        return pd.Series(ts, index=pd.date_range(
            '{}-{:0>2d}-01'.format(int(tsp[0]), round((tsp[0] % 1) * 12 + 1)), periods=round((tsp[1] - tsp[0]) * tsp[2]), 
            freq=({12.0: 'M', 4.0: 'Q', 1.0: 'Y'}[tsp[2]] + ['S', ''][period_end])
        ).map(lambda x: x.replace(day=1) if not month_end else (x.replace(day=1) + relativedelta.relativedelta(months=1) - datetime.timedelta(days=1))))
    warn('Unknown index type: start={}, end={}, freq={}'.format(*[x for x in robjects.r.tsp(ts)]))
    return pd.Series(ts)

def zoo2series(ts):
    return pd.Series(ts, pd.to_datetime(list(base.attr(ts, 'index')), unit='d'))

#
# ceic
# * mirror to ceic library funtions in R
##
for i in dir(ceic):
    if re.match('ceic_', i):
        globals()[i[5:]] = getattr(ceic, i)

#
# ceic.search_all
# * similar to ceic.search in R, but extract all search result
##
def search_all(*args, **kwargs):
    result = ceic.ceic_search(*args, **kwargs)
    while base.nrow(result)[0] < base.attr(result, 'total')[0]:
        result = base.rbind(result, ceic.ceic_next(result))
    return rcv.rpy2py(result)

def search(*args, **kwargs):
    return rcv.rpy2py(ceic.ceic_search(*args, **kwargs))

def metadata(*args, **kwargs):
    return rcv.rpy2py(ceic.ceic_metadata(*args, **kwargs))

def timepoints(*args, index='auto', **kwargs):
    _raise('Wrong value for index param. ') if not index in ('auto', 'date', 'numeric') else None
    prev_format = base.getOption('ceic.format')
    ceic.ceic_format('zoo' if index == 'date' else 'ts')
    ts = ceic.ceic_timepoints(*args, **kwargs)
    ceic.ceic_format(prev_format)
    if index == 'date':
        return zoo2series(ts)
    else:
        return ts2series(ts, index=index)

def series(*args, index='auto', **kwargs):
    _raise('Wrong value for index param. ') if not index in ('auto', 'date', 'numeric') else None
    prev_format = base.getOption('ceic.format')
    ceic.ceic_format('zoo' if index == 'date' else 'ts')
    result = ceic.ceic_series(*args, **kwargs)
    ceic.ceic_format(prev_format)
    try:
        result = [(zoo2series(ts) if index == 'date' else ts2series(ts, index=index), pd.Series({i: v[0] for (i, v) in meta.items()})) for (ts, meta) in result]
    except ValueError:
        ts, meta = result
        result = [(zoo2series(ts) if index == 'date' else ts2series(ts, index=index), pd.Series({i: v[0] for (i, v) in meta.items()}))]
    return result
    

def _raise(msg):
    raise Exception(msg)

#
# Panel class
# * for easy extraction of panel data as wide dataframe structure
# * so far, freq='Y', 'Q' and 'M' are supported, while 'H', 'W' and 'D' are not supported
##
class Panel:
    multipliers = {
        'NA': 1,   'TH': 1e3, 'MN': 1e6, 'BN': 1e9, 'TN': 1e12, 
        'TT': 1e4, 'QN': 1e15, 'HM': 1e8, 'HT': 1e5, '2T': 2e4, 
        'HB': 1e11, 'TM': 1e7, 'TB': 1e10
    }
    coco_plus = pd.DataFrame({'name_short': ['Euro Area', 'European Union'], 'name_official': ['Euro Area', 'European Union'], 'regex': ['euro.*(?=area|zone)', 'european.*union'], 'ISO3': ['_EZ', '_EU']})
    #geo = pd.read_csv('ceic_geo.csv')
    freq_names = {'Y': ['year'], 'Q': ['year', 'quarter'], 'M': ['year', 'month'], None: ['date']}
    freq_groupby = {'Y': lambda d: d.year, 'Q': [lambda d: d.year, lambda d: d.quarter], 'M': [lambda d: d.year, lambda d: d.month]}
    
    def __init__(self, country_list=None, freq=None):
        self.df_wide = pd.DataFrame()
        self.df_narrow = pd.DataFrame()
        self.df_variables = pd.DataFrame()
        self.df_metadata = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], []], names=['variable', 'country_iso']))
        self.country_list = country_list
        _raise('Only freq = \'Y\', \'Q\', \'M\' and None supported') if not freq in ('Y', 'Q', 'M', None) else None
        self.freq = freq
        return
    
    def search_tickers(self, var_name, *args, unique=False, **kwargs):
        # CEIC search
        result = search_all(*args, **kwargs)
        _raise('The CEIC search result does not give unique countries') if unique and result.duplicated(subset='country').any() else None
        result = result[result['subscribed'] == 'true']
        result.drop_duplicates(subset='country', keep='first', inplace=True)
        result['country_iso'] = self._country2iso(list(result['country']))
        result['variable'] = var_name
        if type(self.country_list) == list:
            result = result[result['country_iso'].isin(self.country_list)]
        result.set_index(['variable', 'country_iso'], inplace=True)
        # update df_metadata and df_variables
        for (i, row) in result.iterrows():
            self._set_ticker_meta(row['id'], var_name, i[1], row)
    
    def _set_ticker_meta(self, ticker, var_name, country_iso, meta):
        self.df_variables.loc[country_iso, var_name] = str(int(ticker))
        self.df_metadata[(var_name, country_iso)] = meta
        warn('Series [{}, {}] with ticker {} has status of {}'.format(var_name, country_iso, ticker, meta['status'])) if meta['status'] != 'Active' else None
    
    def add_ticker(self, ticker, var_name, country_iso=None):
        meta = metadata(ticker).iloc[0]
        country_iso = self._country2iso(meta['country']) if country_iso == None else country_iso
        _set_ticker_meta(ticker, var_name, country_iso, meta)
    
    def _country2iso(self, *args):
        return coco.convert(*args, to='ISO3', additional_data=self.coco_plus)
    
    def add_tickers(self, mapper):
        tickers = list(set([ i for v in mapper.values() if isinstance(v, list) for i in v ] + [ v for v in mapper.values() if not isinstance(v, list)]))
        meta = metadata(tickers)
        # for those { 'var': [...tickers...] }
        for (var_name, tickers) in mapper.items():
            if not (isinstance(var_name, tuple) or isinstance(var_name, list)):
                for t in tickers:
                    m = meta[meta['id'] == str(int(t))].iloc[0]
                    self._set_ticker_meta(str(int(t)), var_name, self._country2iso(m['country']), m)
        # for those { ('var', 'iso'): ticker }
        for (var_iso, t) in mapper.items():
            if isinstance(var_iso, tuple) or isinstance(var_iso, list):
                m = meta[meta['id'] == str(int(t))].iloc[0]
                self._set_ticker_meta(str(int(t)), *var_iso, m)
    
    def rebase_all(self, num_attempts=10):
        for (var_iso, meta) in self.df_metadata.iteritems():
            if meta['status'] == 'Rebased':
                meta_new = meta.copy()
                for i in range(num_attempts):
                    if meta_new['status'] == 'Rebased' and meta_new['replacements']:
                        meta_new = metadata(meta_new['replacements']).iloc[0]
                    elif meta_new['status'] != 'Rebased':
                        break
                self._set_ticker_meta(meta_new['id'], *var_iso, meta_new)
                (warn if meta_new['status'] != 'Active' else print)('Series [{}, {}] with {} rebased to {} whose status is {}'.format(var_iso[0], var_iso[1], meta['id'], meta_new['id'], meta_new['status']))
    
    def extract(self, unitMultiplier=True, agg={}, spec_agg={}):
        _raise('agg and spec_agg should be a dictionary') if type(agg) != dict or type(spec_agg) != dict else None
        s_narrow = []
        # ss = { id: series_index_by_datetime }
        ss = series(list(set(self.df_variables.melt().dropna()['value'])), index='date')
        ss = {meta['id']: ts * (self.multipliers[meta['multiplierCode']] ** unitMultiplier) for (ts, meta) in ss}
        # iterate through df_variables and find the corresponding series
        for (i, col) in self.df_variables.iteritems():
            for (c, ticker) in col.iteritems():
                if ticker in ss:
                    s = ss[ticker].copy()
                    s.index.name = 'date'
                    if self.freq != None:
                        s = s.groupby(self.freq_groupby[self.freq]).agg(spec_agg.get((i, c), agg.get(i, 'last')))
                        s.index.rename(self.freq_names[self.freq] if self.freq != 'Y' else 'year', inplace=True)
                    s_narrow.append(s.reset_index().assign(country_iso=c, variable=i))
        # concat to narrow form and unstack to wide form
        self.df_narrow = pd.concat(s_narrow).set_index(['country_iso', *self.freq_names[self.freq], 'variable'])[0].dropna()
        self.df_wide = self.df_narrow.unstack()
    
    def splice(self, mapper, method, back=True):
        _raise('wrong value for method') if not method in ('ratio', 'level') else None
        p = Panel(self.country_list, self.freq)
        p.add_tickers(mapper)
        p.extract()
        df_old = (p if back else self).df_narrow.unstack(('variable', 'country_iso'))
        df_new = (self if back else p).df_narrow.unstack(('variable', 'country_iso'))
        df_adj = df_new.combine(df_old, lambda s_n, s_o: s_n / s_o).mean() * df_old if method == 'ratio' else df_new.combine(df_old, lambda s_n, s_o: s_n - s_o).mean() + df_old
        df_new = df_new.combine_first(df_adj)
        self.df_narrow = df_new.stack(('variable', 'country_iso')).reset_index().set_index(['variable', *self.freq_names[self.freq], 'country_iso'])[0].dropna()
        self.df_wide = self.df_narrow.unstack()