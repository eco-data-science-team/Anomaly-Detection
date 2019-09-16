# -*- coding: utf-8 -*-

"""
This class uses the OSI soft provided python library to access the PI web api
It helps to wrap the osisoft api functions into more usable functions for the
typical workflow of the ACE team

@author: Correy Koshnick <ckoshnick@gmail.com>

"""

# =============================================================================
# --- Imports
# =============================================================================

import os
import sys
import time
import json
import logging
import requests

import pandas as pd

cwd = os.getcwd()
if cwd[-3:] == 'src':
    config_path = cwd
else:
    config_path = cwd + '/src' 
sys.path.append(config_path)
from collections import defaultdict
from requests_kerberos import HTTPKerberosAuth, OPTIONAL
from requests_kerberos.exceptions import KerberosExchangeError
from osisoft.pidevclub.piwebapi.rest import ApiException
from osisoft.pidevclub.piwebapi.pi_web_api_client import PIWebApiClient
from osisoft.pidevclub.piwebapi.models import PIPoint, PIStreamValues, PITimedValue

# Needed for the password storage
local_path = os.path.dirname(__file__)

__all__ = ['pi_client', 'PI_builder']


class PiClient(object):
    """ Custom layer between the user and the OSIsoft Python PI web API library
    The library can be found here:

    https://github.com/osimloeff/PI-Web-API-Client-Python
    # as of Dec 2018 the library is down, but stored in ecotools private repo

    """

    def __init__(self,
                 root="https://util-cs-iis.ou.ad3.ucdavis.edu/piwebapi/",
                 data_server_path="\\\\UTIL-PI-P",
                 asset_server_path="\\\\UTIL-AF",
                 asset_database='Gimli',
                 time_zone='America/Los_Angeles',
                 use_kerberos=True,
                 username=None,
                 password=None):
        """
        Parameters
        ----------
        root - str
            PI web API root url
            -Read / write: https://util-cs-iis.ou.ad3.ucdavis.edu/piwebapi/
            -Read only: https://ucd-pi-iis.ou.ad3.ucdavis.edu/piwebapi/

        data_server_path - str
            Name of data server to access within PI

        asset_server_path - str
            Name of the asset server in PI

        asset_database - str
            Name of the database in the asset_server

        time_zone - str
            timezone name from TZ database https://www.iana.org/time-zones

        use_kerberos - Bool
            True - use the windows authentication for PI authentication
            False - Use a username/password basic authentication

        username - str or None or 'file'
            The users UCD email to authenticate in PI with
            If only he username is provdided, the user will be prompted for
            a password, and the password will be stored locally but as
            base64 encoded text

            to load the stored UN:PW data use username = 'file'

        password - str or list of ordinals or None
            If str, try to authenticate with the username
            if list, the list will be converted element wise by chr(x) and
            joined into a single string
            if None and use_kerberos = False, then prompt user for password

        Returns
        -------
        class instance (pi)
        """

        self.default_web_id_source = 'generate'
        self.time_zone = time_zone

        # Athentication Section
        # Allows us to access the read only instance of the API if we want to
        # avoid authentication all together
        if root.lower() in ['readonly', 'read only', 'r', 'ro', 'read-only']:
            root = "https://ucd-pi-iis.ou.ad3.ucdavis.edu/piwebapi/"
            use_kerberos = False
            username = 'pass'
            password = 'pass'
            self.default_web_id_source = 'get'

        elif username is not None:

            use_kerberos = False

            if isinstance(password, list):
                password = ''.join([chr(x) for x in password])

            if username == 'file':
                username, password = self._load_user_file()

            else:
                password = input(f"{username}, please provide your password\n>>")
                save_pw = input("save password? [y/n]\n>>")
                if save_pw == 'y':

                    with open(os.path.join(local_path, 'file.txt'), 'w+') as f:
                        save_string = self._xor_crypt_string(f'{username}:{password}')
                        f.write(save_string)

        self.client = PIWebApiClient(root, useKerberos=use_kerberos,
                                     username=username,
                                     password=password,
                                     verifySsl=True)

        try:  # Attempt to test connection
            # OPPM dataServer object
            self.dataServer = self.client.dataServer.get_by_path(
                    data_server_path)

        except Exception as e:
            print(e)
            username, password = self._load_user_file()
            self.client = PIWebApiClient(root, useKerberos=False,
                                         username=username,
                                         password=password,
                                         verifySsl=True)
            try:
                # OPPM dataServer object
                self.dataServer = self.client.dataServer.get_by_path(
                    data_server_path)
            except:
                print("FAILED TO AUTHENTICATE!! Returning None")
                return

        # /End authentication section

        asset_database_path = asset_server_path + '\\' + asset_database
        # OPPM assetDatabase object
        self.assetDatabase = self.client.assetDatabase.get_by_path(
                asset_database_path)

    # =========================================================================
    # --- Utilities
    # =========================================================================

    def _timestamp(self):
        """ Get the current time, for print() monitoring of process """
        return time.strftime('%H:%M:%S', time.localtime())

    def _xor_crypt_string(self, data, key='bWF0dF9pc19zb19zYWx0eQ=='):
        if len(data) <= len(key):
            pass
        elif len(data) > len(key):
            while len(key) < len(data):
                key += key

        key = key[0:len(data)]

        return bytearray(a^b for a, b in zip(*map(bytearray, [data, key], ['utf-8']*len(data)))).decode('utf-8')

    def _load_user_file(self):
        try:
            with open(os.path.join(local_path, 'file.txt'), 'r') as f:
                line = ''.join(f.readlines())
                line = self._xor_crypt_string(line)
                username, password = line.split(':')
                return username, password

        except FileNotFoundError:
            print('You must save your login before loading it!')
            return (None, None)

    def _get_web_ids(self, tags, _selected_fields=['webid', 'name']):
        """ This function accepts PI point names and returns a dictionary
        containing the tag:webID  key:value pairs. All ids are obtained by
        querying the PI server (slow)

        Parameters
        ----------

        tags - list of str or str
            Each item in tags should be a PIpoint. if a single tag as a string
            it is converted to 1 item list for itertation
            if any of the tags are invalid it is dropped from the list
        _selected_fields = list of str
            The values that are returned in the PI web api response
            For this function webid, and name are sufficient

        Returns
        -------
        tag_dict - dict
            tag:web_id pairs
        """

        tag_dict = {}

        print("Grabbing WebIDs ...")
        if isinstance(tags, list):
            pass
        if isinstance(tags, str):
            tags = [tags]

        # Generates the 'Items.webid;Items.name' string
        fields = "Items." + ";Items.".join(_selected_fields).strip(';.')

        for tag in tags:
            resp = self.client.dataServer.get_points(
                    self.dataServer.web_id,
                    max_count=None,
                    name_filter=tag,
                    selected_fields=fields,
                    start_index=None,
                    web_id_type=None)

            # Handle strings that are not PI points
            try:
                tag_dict[tag] = resp.items[0].web_id
            except IndexError:
                print('Warning: Tag {} not found in dataServer'.format(tag))

        print("WebID collection complete!")
        return tag_dict

    def _generate_webids(self, tags, output='list'):
        """ Convert a list of tags into webids client side (fast)

        Parameters
        ----------
        tags - list of str or str
            List of PI tags, or a single PI tag

        output - str
            list - returns only list of web_ids
            dict - returns tag:web_id pairs
        """
        # handles pd.columns too
        if isinstance(tags, pd.core.indexes.base.Index):
            tags = list(tags)

        if isinstance(tags, str):
            tags = [tags]

        def convert(tag):
            return self.client.webIdHelper.generate_web_id_by_path(
                    f"\\\\{self.dataServer.name}\\{tag}", type(PIPoint()))

        if output == 'list':
            return [convert(tag) for tag in tags]
        if output == 'dict':
            return {tag: convert(tag) for tag in tags}

    def _parse_stream(self, piStreamValues,
                      calculation='interpolated',
                      interval='1h',
                      single_time_axis=True,
                      _drop_duplicates=True,
                      _duplicate_rule='last',
                      _convert_cols=True):
        """ Parse the piStreamValues object into a pd.DataFrame. The parsing
        method is dependent on the caluclation type


        Parameters
        ----------
        piStreamValues - piStreamValues
            Takes a piStreamValues object which is an object in the oppm module
        calculation - str
            Different calculation methods require different parsing options

        interval - str
            Specifies the size of the time interval the data should be reported
            in. Only needed for the 'manual' resampling in "calculated"

        single_time_axis - bool
            Converge all of the individual dataframes on a single time axis

        convert_cols - bool
            Whether or not to convert all of the columns to numeric type. This
            will replace any non numeric value with nan


        Returns
        -------
        df - pd.DataFrame
            The joined dataframe of all parsed dataframes with a single ts axis

        dfCollector - list(pd.DataFrame)
            - a list of individual dataframes
        """

        # dfCollector will hold all dataframes until single pd.concat() call
        dfCollector = []

        if len(piStreamValues.items) == 0:
            print('piStreamValues is empty!')
            return pd.DataFrame()

        for stream in piStreamValues.items:

            if len(stream.items) == 0:
                print('Failed to construct df on {}. '
                      'The returned data is Null or None'.format(stream.name))
                continue

            if calculation == 'summary':
                df = self.client.data.convert_summary_to_df(
                        stream.items, selected_fields=['timestamp',
                                                       'value'])
                df.drop('SummaryType', inplace=True, axis=1)
            else:
                df = self.client.data.convert_to_df(
                        stream.items, selected_fields=['timestamp',
                                                       'value'])

            # Reaplce generic "Value" column with tag name
            df.columns = df.columns.str.replace('Value', stream.name)

            # Convert timestamp and set as index
            df['Timestamp'] = pd.to_datetime(
                    df['Timestamp'], infer_datetime_format=True)
            df = df.set_index('Timestamp', drop=True)

            # Fix timezone
            df = self._utc_to_local(df, self.time_zone)

            # Removes the str, blanks and dictionary entries
            if _convert_cols:
                df = df.apply(pd.to_numeric, errors='coerce')

            if calculation == 'calculated':
                # Resample recorded data to specified inverval
                df = df.resample(interval).mean()

            # Keep last entry if duplicate values for one timestamp
            if _drop_duplicates:
                df = df.loc[~df.index.duplicated()]

            dfCollector.append(df)

        if single_time_axis:
            try:
                df = pd.concat(dfCollector, axis=1, join='outer')
                return df
            except ValueError:
                print("ValueError: No objects to concatenate -- chunk empty")
                return pd.DataFrame()

        else:
            print("Parse complete!")
            return dfCollector

    def _utc_to_local(self, data, local_zone=None):
        """ Function takes in pandas dataframe generated from PI data stream
        and adjusts index according to match local timezone

        Parameters
        ----------
        data: Dataframe
            pandas dataframe of json timeseries response from server

        local_zone: string
            pytz.timezone string of specified local timezone to change index to

        Returns
        -------
        data: Dataframe
            Pandas dataframe with timestamp index adjusted for local timezone
        """
        # Grab timezone from class attributes - Instantiate your class with TZ
        if local_zone is None:
            local_zone = self.time_zone

        # accounts for localtime shift

        data.index = data.index.tz_localize(
                "UTC", ambiguous='NaT', errors='coerce').tz_convert(local_zone)
        # HACK this needs to be improved with nonexistent
        data = data.loc[pd.notnull(data.index)]
        # Gets rid of extra offset information so can compare with csv data
        data.index = data.index.tz_localize(None)

        return data

    def _local_to_utc(self, data, local_zone=None):
        """ Function takes in pandas dataframe generated locally and adjusts
        index to match UTC to be sent to PI (always in UTC)

        Parameters
        ----------
        data: Dataframe
            pandas dataframe of json timeseries response from server

        local_zone: string
            pytz.timezone string of specified local timezone to change index to

        Returns
        -------
        data: Dataframe
            Pandas dataframe with timestamp index adjusted for UTC timezone
        """
        # Grab timezone from class attributes - Instantiate your class with TZ
        if local_zone is None:
            local_zone = self.time_zone

        # Necessary to assign local zone before converting
        # TODO Upgrade pandas and add nonexistant keyward
        data.index = data.index.tz_localize(local_zone,
                                            ambiguous='NaT',
                                            errors='coerce').tz_convert(None)
        # HACK this needs to be improved with nonexistent
        data = data.loc[pd.notnull(data.index)]
        # Gets rid of extra offset information so can compare with csv data
        data.index = data.index.tz_localize(None)

        return data

    def _verify_point_source(self, tag, source='webapi'):
        """ Returns True if the pointsource of tag matches source """

        try:
            pointsource = self.get_point_attribute(tag=tag,
                                                   attribute='pointsource')
        except ApiException as e:
            if e.status == 404:
                return 'No Tag'
            else:
                return f'unknown error in verify point source {e}'

        return pointsource == source

    def _verify_tags(self, tags):
        """ Verifies several tags pointsource is 'webapi' to safeguard data
        in tags that are not webapi

        Returns
        -------
        pd.DataFrame
            A df showing the verification result for each tag

        """
        verifyDict = {}

        if isinstance(tags, pd.core.indexes.base.Index):
            tags = list(tags)

        for tag in tags:
            verifyDict[tag] = {
                    'verified': self._verify_point_source(tag,
                                                          source='webapi')
                    }

        return pd.DataFrame.from_dict(verifyDict, orient='index')

    def _chunker(self, iterable, chunk_size):
        """ Split iterable into smaller chunks as generator """
        for i in range(0, len(iterable), chunk_size):
            yield iterable[i:i+chunk_size]

    # =========================================================================
    # --- Searching
    # =========================================================================

    def search_by_point(self, search_query, output='tags', _returnRaw=False):
        """ Searches for PI points in data server that match the search_query
        string. Supports wildcarding

        Parameters
        ----------
        search_query - str or list
            Search string to find PI tags. eg. ACAD*AHU*Outside Air Temp*

        output - str
            'tags' - Output list of tag names
            'webids' - Output list of webids only
            'both' - Output dict of tag:webid pairs


        Returns
        -------
        search_result - list / dict

        """

        if isinstance(search_query, str):
            search_query = [search_query]
        elif isinstance(search_query, list):
            pass
        else:
            raise ValueError('search_query must be a str, or list of str')

        output_options = {'tags': [], 'webids': [], 'both': {}}
        if output not in list(output_options.keys()):
            print(f'output= must appear in {output_options}')

        search_result = output_options[output]

        for search in search_query:

            resp = self.client.dataServer.get_points(
                    self.dataServer.web_id,
                    max_count=100000,
                    name_filter=search,
                    selected_fields='Items.name;Items.Webid;Items.path;Items.id',
                    start_index=None,
                    web_id_type=None).items

            if _returnRaw:
                return resp

            for element in resp:
                if output == 'tags':
                    search_result.append(element.name)

                if output == 'webids':
                    search_result.append(element.web_id)

                if output == 'both':
                    search_result[element.name] = element.web_id

        if len(search_result) == 0:
            print("Warning: No point found with "
                  "search query {}".format(search_query))
            return []

        return search_result

    def group_tags(self, all_tags, parent_level, sensor_group, sep='.'):
        """
        This function will filter a list of tags to make sure a complete set of
        points are available. This helps when pulling data from PI to create an
        analysis of multiple points. If points A,B,C are being compared but
        point B does not exist, it will elimate that system from the group so
        time is not wasted pulling data for points A and C.

        The function operates by ensuring that a particular level in the name
        hierarchy (the parent_level) has associated with it the points listed
        in sensor_group.

        Consider the following example:

        If we want to check the recirculation of office AHUs we need the
        Return Air Temp, Mixed Air Temp, and Supply Air Temp. In the following
        case AHU01 is a lab AHU and does not have a Return Air Temp, so we dont
        need to analyze this AHU. If we filter that from our group of tags to
        pull data we would run the function as:

        group_tags(all_tags, parent_level=2, sensor_group=['Mixed Air Temp',
                                                      'Return Air Temp',
                                                      'Supply Air Temp'])

         u'BLDG.AHU.AHU01.Supply Air Temp',
         u'BLDG.AHU.AHU01.Supply Air Temp Setpoint',

         u'BLDG.AHU.AHU02.Mixed Air Temp',
         u'BLDG.AHU.AHU02.Return Air Temp',
         u'BLDG.AHU.AHU02.Supply Air Temp',
         u'BLDG.AHU.AHU02.Supply Air Temp Setpoint',

        AHU01 is dropped because it does not have all 3 sensors, and AHU02
        Supply Air Temp Setpoint is also dropped because it is not part of the
        sensor_group

         output = [u'BLDG.AHU.AHU02.Mixed Air Temp',
                   u'BLDG.AHU.AHU02.Return Air Temp',
                   u'BLDG.AHU.AHU02.Supply Air Temp',]

        NOTE: This function has nothing to do with missing data or data quality

        Parameters
        ----------
            all_tags - list of str
                The entire pool of tags that may be pulled
                This list needs to be pregenerated by a function like
                search_tags()
            parent_level - int
                The index of the repeated name within the tag hierarchy
            sensor_group - list of str
                The sensors that need to be included in each grouping of the
                parent level
            sep - str
                The seperator of the levels in the PI point name

        Returns
        -------


        """

        # Initial Vars
        ddict = defaultdict(list)
        pass_value = len(sensor_group)
        wrong_list, short_list = [], []

        # Group tags by splitting, and using the parentName as dict Key
        for tag in all_tags:
            splitTag = tag.split(sep)
            if splitTag[-1] in sensor_group:
                ddict['.'.join(splitTag[0:parent_level])].append(tag)

            else:
                wrong_list.append(tag)

        # Check which parents have the proper # of children
        for parent_key in ddict:
            if len(ddict[parent_key]) < pass_value:
                short_list.append(parent_key)

            else:
                pass
        # Delete Failures from defaultdict
        for key in short_list:
            del ddict[key]

        # Display which points were discarded and why
        print('Points with the wrong sensors')
        print(wrong_list)
        print()
        print('Correct sensors, but not a complete set')
        print(short_list)

        # Re-combine items in dict into a flat list
        result_list = []
        for key, value in ddict.items():
            result_list += value

        return sorted(result_list)

    # =========================================================================
    # ---Reading
    # =========================================================================

    def get_point_attribute(self, tag=None, webid=None, attribute=None):
        """ Return the value of given attribute. Pass in either the name
        of the point or the webid of the point """

        if tag:
            webid = self._generate_webids(tag)[0]

        if webid:
            resp = self.client.point.get_attribute_by_name(attribute,
                                                           webid)
            return resp.value

    def get_stream_by_point(self,
                            tags,
                            start='y',
                            end='t',
                            calculation='interpolated',
                            interval='15m',
                            _max_count=None,
                            _weight='TimeWeighted',
                            _summary_type='average',
                            _chunk_size=20,
                            _convert_cols=True,
                            _drop_duplicates=True,
                            _web_id_source=None,
                            _buffer_time=2,
                            _max_timeouts=4):

        """ Pull 1 or many tags through the PI web API and returns a dataframe

        The dataframe is a concatenation of many dataframes on the timeseries
        index with an outer default.


        Parameters
        ----------
        tags - str / list / dict
            str and list should be a list or string of PI tags
            dict should be {tag:webid} pairs

        start - str
            starting date for query of the date fmt yyyy-mm-dd

        end - str
            ending date for query of the date fmt yyyy-mm-dd

        calculation - str
            'interpolated' – This performs a linear interpolation to
                estimate the value of the point at that time based on
                surrounding raw data points.

                For time stamps before the first recorded value,
                the function returns either Pt Created or No Data.

                For time stamps between two recorded values, the function
                determines the value at the time stamp using linear
                interpolation between the recorded values. For points that
                store discrete values, such as digital state points or step
                points, the function returns the last recorded value that
                precedes the time stamp.

                For time stamps after the last recorded value, the returned
                value depends on the point type:

                    For historical PI points, the function returns the most
                    recent value.

                    For future PI points, the function returns No Data.

            'summary'  - This will take the average of all recorded values
            that exist within in time interval. If no values exists, nan will
            be returned.

            'calculated' – Same functionality as above, except the recorded
            data is transferred to the local client and pandas does the
            resampling and averaging. Performance may vary between the two
            functions depending on the data, and size.

            'recorded' – raw data and raw time stamps. If pulling multiple
            columns, the index will grow for all columns to accommodate each
            individual timestamp and nan values will be filled in for the
            mismatches.

        interval - str
            Specifies the size of the time interval the data should be reported
            in.

        _max_count - int
            The maximum number of items returned from PI. This will be divided
            among each tag, and is most important to modify with 'recorded'
            and 'calculated' In these cases it is best to set _max_count to
            the PI server max_count of 1,500,00 / chunk_size. If this fails to
            retrieve every value, change the chunk_size

        _chunk_size=50 - int
            The number of tags to include in each URL request. This is needed
            to avoid PI server timeout errors, and URL to long errors. A good
            range for sizes is 1-50. For longer data pulls, use smaller chunks

        _convert_cols - bool
            Convert columns from object to numeric type, default is to coerce
            errors

        _drop_duplicates - bool
            Drops any timestamps with same values, keeping the "last" option

        _web_id_source - str
            Options are 'generate' and 'get'
            'get' querys the PI server for the web_id of each tag, this is
            needed for tags that may not exist. Sending a generated web_id to
            PI that does not exist will cuase the entire chunk to fail
            'generate' can be used anytime the all 100% of the tags are sure to
            exist.

        _buffer_time - int
            The number of seconds between subsequent chunks being sent, this
            parameter can help large data pulls by giving the server a short
            break

        _max_timeouts - int
            the maximum number of timeout_failures that are allowed to occur
            before the program moves to the next chunk. This is an important
            health parameter to prevent any client from eternally querying PI

        _summary_type - str
            The statistical calculation that will be applied during to the data
            when requesting calculation='summary'. The potential calcs are here
            https://techsupport.osisoft.com/Documentation/PI-Web-API/help/topics/summary-type.html

        _weight - str
            The type of weighting applied to the datapoints when requesting
            data of type calculation='summary'. The weighting documention:
            https://techsupport.osisoft.com/Documentation/PI-Web-API/help/topics/calculation-basis.html

        Returns
        -------
        df - pd.DataFrame
            The data for each tag pulled from PI
        """

        if len(tags) == 0:
            raise ValueError("tags can not be empty")

        if _max_count is None:
            _max_count = round(1500000 / len(tags))

        _sum_types = ['average', 'minimum', 'maximum', 'range', 'stddev',
                      'count', 'percentgood', 'max', 'total', 'none', 'all']
        if _summary_type not in _sum_types:
            raise ValueError(f'_summary_type must be one of {_sum_types}.')

        # Web_id Section
        if isinstance(tags, dict):
            web_ids = list(tags.values())

        elif isinstance(tags, list) or isinstance(tags, str):

            if _web_id_source is None:
                _web_id_source = self.default_web_id_source

            # Assume list is list of strings (tags) and you must get webids
            if _web_id_source == 'generate':
                web_ids = self._generate_webids(tags)
            elif _web_id_source == 'get':
                web_ids = list(self._get_web_ids(tags).values())
            elif _web_id_source == 'webid':
                web_ids = tags
            else:
                raise ValueError(f'_web_id_source must be "generate" or "get"')
        else:
            raise ValueError(f'tags must be of type str, dict, list, not {type(tags)}')

        # Chunk handling
        if len(web_ids) == 1:
            chunks = 1
        elif _chunk_size == 1:
            chunks = len(web_ids)
        else:
            chunks = int(len(web_ids)/_chunk_size) + 1
            if len(web_ids) % _chunk_size == 0:
                chunks -= 1

        dfCollector = []  # Holds the chunks

        # Data pulling section
        for i in range(chunks):
            try:
                web_id_chunk = web_ids[i * _chunk_size:(i + 1) * _chunk_size]
            except IndexError:
                web_id_chunk = web_ids[i * _chunk_size:len(web_ids)]

            # Timeout while loop
            timeout_failures = 1
            while timeout_failures <= _max_timeouts:
                if i > 0:
                    time.sleep(_buffer_time)
                print("{}: Sending API request... Chunk {} of {}"
                      "".format(self._timestamp(), i+1, chunks))

                # Start time for elapsed calculation (each chunk)
                start_counter = time.time()

                try:  # API call

                    if calculation == 'recorded':

                        piStreamValues = self.client.streamSet.get_recorded_ad_hoc(
                                web_id_chunk, start_time=start, end_time=end,
                                max_count=_max_count,
                                time_zone=self.time_zone)

                    elif calculation == 'interpolated':

                        piStreamValues = self.client.streamSet.get_interpolated_ad_hoc(
                                web_id_chunk, start_time=start, end_time=end,
                                interval=interval,
                                time_zone=self.time_zone)

                    elif calculation == 'calculated':

                        piStreamValues = self.client.streamSet.get_recorded_ad_hoc(
                            web_id_chunk, start_time=start, end_time=end,
                            max_count=_max_count,
                            time_zone=self.time_zone)

                    elif calculation == 'summary':

                        piStreamValues = self.client.streamSet.get_summaries_ad_hoc(
                            web_id_chunk, start_time=start, end_time=end,
                            summary_duration=interval,
                            summary_type=[_summary_type],
                            calculation_basis=_weight,
                            time_zone=self.time_zone)

                except ApiException as e:
                    # Search for error string in body text
                    if e.body.find("is greater than the maximum allowed") > 0:
                        print("Your search exceded the PI server max_count; "
                              "try reducing the _chunk_size")
                        piStreamValues = PIStreamValues(items=[])
                        raise ValueError('Exceded max size')

                    elif e.status == 500:
                        print(e.error)
                        piStreamValues = PIStreamValues(items=[])
                        break

                    else:
                        print("Encountered an unhandled HTTP error!")
                        print(e)
                        piStreamValues = PIStreamValues(items=[])
                        break

                except Exception as e:
                    print("WARNING! ! Encountered an unhandled error!")
                    print(e)
                    piStreamValues = PIStreamValues(items=[])
                    break

                elapsed_time = time.time() - start_counter

                print(f"{self._timestamp()}: Response recieved for "
                      f"{len(piStreamValues.items)} tags! ({round(elapsed_time, 2)})")

                # Timeout failure checking
                if len(piStreamValues.items) != len(web_id_chunk) and elapsed_time >= 59:

                    # If failed: make chunk n smaller, try again
                    web_id_chunk = web_id_chunk[len(piStreamValues.items):]

                    dfCollector.append(self._parse_stream(
                        piStreamValues, calculation=calculation,
                        interval=interval, single_time_axis=True,
                        _convert_cols=_convert_cols))

                    print(f'{self._timestamp()}: Stream timeout error! '
                          f'attempting again! {timeout_failures} / {_max_timeouts}')

                    if timeout_failures >= _max_timeouts:
                        print(f'{self._timestamp()}: Too many failures!')
                        raise TimeoutError('PI request failed too many times')
                    else:
                        timeout_failures += 1
                else:
                    break

            # Hold all chunks in a list
            dfCollector.append(self._parse_stream(
                    piStreamValues, calculation=calculation,
                    interval=interval, single_time_axis=True,
                    _drop_duplicates=_drop_duplicates,
                    _convert_cols=_convert_cols))

        # Join all chunks into single df
        df = pd.concat(dfCollector, axis=1, join='outer')

        return df

    # =========================================================================
    # --- Writing
    # =========================================================================

    def _write_series_to_stream(self, series, web_id):
        """ Convert pd.Series object to oppm PIStreamValues by iterating over
        all rows in series.

        requires series to have a timeseries like index

        """

        # Instantiate object
        PIStream = PIStreamValues(web_id=web_id)
        # Populate object's .items attribute (for sending to PI)
        PIStream.items = [PITimedValue(timestamp=index, value=value) for
                          index, value in series.dropna().items()]

        return PIStream

    def _write_df_to_stream(self, df):
        """ Writes all columns of dataframe to PITimedValue, packages them in
        PIStreamValues and posts them to PI via
        self.client.streamSet.update_values_ad_hoc

        """

        # Hacky way to allow single columns to be passed in. This might break
        if isinstance(df, pd.Series):
            df = df.to_frame()

        df = self._local_to_utc(df)

        # Package df as dictionary of web_id:series pairs
        series_dict = {}
        web_ids = self._generate_webids(list(df.columns), output='dict')

        for col in df.columns:
            # map column name to web_id - series of data
            series_dict[web_ids[col]] = df[col]
        # NOTE this seems a little backwards of the tag:webid paradigm. Should
        # it be corrected? It is working as is, just conventionally backwards
        streams = [self._write_series_to_stream(ser, web_id) for
                   web_id, ser in series_dict.items()]

        return streams

    def set_point_attribute(self,
                            tag=None,
                            webid=None,
                            attribute=None,
                            value=None):
        """ Modifies the attribute value of a single point and attribute with
        the input value. Only works on PI points, not for AF attributes

        Parameters
        ----------
        tag - str
            The name of the pi point that will be modified
            will be converted to webid if web id not provided
        webid - str
            the web id of the point that will be modified
        attribute - str
            The attribute that is going to be modified
        value - str or int
            The new value for the attribute

        Returns
        -------
        resp - HTTP response
            The pi web api response

        """

        if webid is None:
            if tag is not None:
                webid = self._generate_webids([tag])[0]

        url = (f'https://util-cs-iis.ou.ad3.ucdavis.edu/piwebapi/points/'
               f'{webid}/attributes/{attribute}')

        headers = {'content-type': 'application/json'}

        print("Sending request...")
        print(url)
        resp = requests.put(url,
                            data=json.dumps(value),
                            headers=headers,
                            verify=True,
                            auth=HTTPKerberosAuth(mutual_authentication=OPTIONAL))
        print('Response recieved! <[{}]>'.format(resp.status_code))

        return resp

    def set_multiple_attributes(self,
                                updateDict):
        """ Mass updater of attributes.

        Must specify the information in the following format

        updateDict = {'tag1': {att1:val1, att2:val2},
                      'tag2': {att1:val1}}

        """

        response_dict = {}

        # Iterate over the tag1, tag2... outer level
        for key, atts in updateDict.items():
            response_dict[key] = {}
            # Iterate over each att:val pair within the tag
            for att, val in atts.items():
                # Send API request
                resp = self.set_point_attribute(tag=key,
                                                attribute=att,
                                                value=val)
                response_dict[key][att] = resp

        return response_dict

    def write_data_to_pi(self, data,
                         update_option='NoReplace',
                         override=None):
        """
        Digests a df by turning it into a list of PIStreamValues and then sends
        it to PI with the supplied options. This function also has a protection
        feature where it will not write any data to tags that don't have the
        point source "webapi"

        Parameters
        ----------
        data - pd.DataFrame or dict of pd.DataFrame (dict not implemented yet)
            The dataframe

        update_option - str
            How the data will be treated when it encounters points that already
            exist in PI.

            Replace:
                Add the value. If values exist at the specified time, one of
                them will be overwritten.
            Insert:
                Add the value, preserving any values that exist at the
                specified time.
            NoReplace:
                Add the value only if no values exist at the specified time.
            ReplaceOnly:
                If a value exists at the specified time, overwrite it. If one
                does not exist, ignore the provided value.
            InsertNoCompression:
                Insert the value without compression.
            Remove:
                Remove the value if one exists at the specified time.

        override - str
            Provide the requested 'passcode' as a kwarg to bypass the input
            prompting

        Returns
        -------
        response - http response

        """

        # Constants
        option_list = ['ReplaceOnly', 'NoReplace', 'Insert', 'Replace',
                       'Remove']

        y = [112, 105, 110, 107, 100, 105, 110, 111, 115, 97, 117, 114]
        _secret_phrase = "".join([chr(x) for x in y])

        if update_option not in option_list:
            raise ValueError("update_option invalid. must be in {}".format(
                    option_list))

        # Lockout section
        if update_option in option_list:

            if override == _secret_phrase:
                print('override passed! Skipping to tag verification {}'
                      .format(update_option.upper()))
                pass
            else:

                print(data.columns)
                i = input('Do you want to {} data in the above tags '
                          .format(update_option.upper()))

                if i.lower() in ['y', 'yes']:
                    pass
                else:
                    print('!! write_streams_to_pi aborted !!')
                    return

                secret = input('enter "secret" passphrase: ')

                if secret == _secret_phrase:
                    pass
                else:
                    print('failed to enter correct phrase')
                    print('!! write_streams_to_pi aborted !!')
                    return

        verified = self._verify_tags(data.columns)
        pass_data = data[list(verified[verified['verified'] == True].index)]

        if len(pass_data.columns) != len(data.columns):
            print('Warning! not all tags were verified "webapi" tags!')
            print(verified[verified['verified'] != True].columns,'\n')

            logging.info('Warning! not all tags were verified "webapi" tags!')
            logging.info(verified[verified['verified'] != True])

        streamValues = self._write_df_to_stream(pass_data)

        response = self.client.streamSet.update_values_ad_hoc_with_http_info(
                streamValues, update_option=update_option
                )

        return response

    def delete_data_from_pi(self, tags, start, end):
        """ Pass in point name and time ranges. Pull recorded data - then push
        recorded data as a update = remove() to remove all data from that range

        Parameters
        ----------
        tags - str / list / dict
            str and list should be a list or string of PI tags
            dict should be {tag:webid} pairs

        start - str
            starting date for query of the date fmt yyyy-mm-dd

        end - str
            ending date for query of the date fmt yyyy-mm-dd

        Returns
        -------
        resp - http response
        """

        # This is redundant, why not just keep it as streams?
        # This is now handled properly when API timeout occurs becuase of the
        # get_stream_by_point timeout handling section

        # get recorded data from point -- must be recorded so timestamp matches
        df = self.get_stream_by_point(tags, calculation='recorded',
                                      start=start, end=end)

        # send recorded data back with update_option = "Remove"
        resp = self.write_data_to_pi(df, update_option='Remove')

        return resp


def PI_builder(point_names,
               point_source='webapi',
               point_type='Float32',
               compressing=1,
               archiving=1,
               future=0,
               step=0,
               engunits='',
               savexlsx=False,):
    """ Helper function to assist in the creation of PI points. If any input
    is not an iterable. It will be converted to a list of len(point_names) with
    N repitions of the single input str, int or float

    Detailed descriptions of all of the following attributes can be found here
    https://livelibrary.osisoft.com/LiveLibrary/content/en/server-v10/GUID-EA52D970-5D4E-44E6-BA4C-08A3F8CDCD8D#addHistory=true&filename=GUID-160E8B37-F323-4800-A6F5-9B9EDEACFFE4.xml&docid=GUID-5843A046-1B1A-4C73-9FAC-309DC6CE7C07&inner_id=&tid=&query=&scope=&resource=&toc=false&eventType=lcContent.loadDocGUID-5843A046-1B1A-4C73-9FAC-309DC6CE7C07

    Parameters
    ----------
    point_names - list of str or pd.DataFrame.index
        The PI point names to be
    point_source - str
        This will be the 'pointsource' attribute of the PI point

    point_type - str
        allowable inputs: Int16, Int32, Float16, Float32, Float64, String,
                      Blob, Timestamp
    compressing - int
        1 = allow PI to compress the data
        0 = do not allow PI to compress the data
    archiving - int
        1 - Must be set to 1 to allow the point to archive data
    future - int
        1 = allow this point to hold data after "now"
        0 = only allow the point to hold data before "now"
    step - int
        1 - treats all data as step functions
        0 - allow data to interpolate between points
    engunits - str
        The units of the point. can be blank.
    save_xlsx - bool
        Whether or not to save the generated df as an excel in the cwd()


    Returns
    -------
    df - pd.DataFrame
        The result of the builder apparatus - can be loaded straight into
        excel's PI builder

    """

    length = len(point_names)

    builders = {'Selected(x)': 'x',
                'Name': point_names,  # replace in below for loop
                'ObjectType': 'PIPoint',
                'Description': 'VOLUMES',
                'digitalset': '',
                'displaydigits': -5,
                'engunits': engunits,  # str
                'future': future,  # binary
                'pointsource': point_source,  # str
                'pointtype': point_type,
                'ptclassname': 'classic',
                'sourcetag': '',
                'archiving': archiving,  # binary
                'compressing': compressing,  # binary
                'compdev': 0.2,
                'compmax': 28800,
                'compmin': 0,
                'compdevpercent': 0.2,
                'excdev': 0.1,
                'excmax': 600,
                'excmin': 0,
                'excdevpercent': 0.1,
                'scan': 1,
                'shutdown': 0,
                'span': 100,
                'step': step,  # binary
                'typicalvalue': 50,
                'zero': 0,
                'datasecurity': 'piadmin: A(r,w) | piadmins: A(r,w) | PIEngineers: A(r,w) | PIAnalysisService: A(r,w) | PIReader: A(r) | PIInterfaces: A(r,w) | PIICU: A(r,w) | PIWorld: A(r)',
                'ptsecurity': 'piadmin: A(r,w) | piadmins: A(r,w) | PIEngineers: A(r,w) | PIAnalysisService: A(r,w) | PIReader: A(r) | PIInterfaces: A(r,w) | PIICU: A(r,w) | PIWorld: A(r)',
                }

    for k, v in builders.items():
        if isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
            builders[k] = [v] * length
        elif isinstance(v, list):
            try:
                assert(len(v) == length)
            except AssertionError():
                raise AssertionError(f'The length of {k}:{v} must match '
                                     '{point_names}')

    df = pd.DataFrame.from_dict(builders, orient='columns')

    df.set_index('Selected(x)', inplace=True, drop=True)

    if savexlsx:
        df.to_excel('PI builder output.xlsx')

    return df


if __name__ == "__main__":
    pi = pi_client()
    tags = pi.search_by_point('aiti*')
    oat = pi.get_stream_by_point('aiTIT4045', start='2019-01-01', interval='1d')
#    print(tags)
