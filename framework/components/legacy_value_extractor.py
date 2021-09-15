"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import re
import json
from collections import defaultdict


class GrailQA_Value_Extractor:
    def __init__(self):
        self.pattern = r"(?:\d{4}-\d{2}-\d{2}t[\d:z-]+|(?:jan.|feb.|mar.|apr.|may|jun.|jul.|aug.|sep.|oct.|nov.|dec.) the \d+(?:st|nd|rd|th), \d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}|\d{2}/\d{4}|[-]*\d+[.]\d+e[+-]\d+|[-]*\d+e[+-]\d+|(?<= )[-]*\d+[.]\d+|^[-]*\d+[.]\d+|(?<= )[-]*\d+|^[-]*\d+)"

    def detect_mentions(self, question):
        return re.findall(self.pattern, question)

    def process_literal(self, value: str):  # process datetime mention; append data type
        pattern_date = r"(?:(?:jan.|feb.|mar.|apr.|may|jun.|jul.|aug.|sep.|oct.|nov.|dec.) the \d+(?:st|nd|rd|th), \d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})"
        pattern_datetime = r"\d{4}-\d{2}-\d{2}t[\d:z-]+"
        pattern_float = r"(?:[-]*\d+[.]*\d*e[+-]\d+|(?<= )[-]*\d+[.]\d*|^[-]*\d+[.]\d*)"
        pattern_yearmonth = r"\d{4}-\d{2}"
        pattern_year = r"(?:(?<= )\d{4}|^\d{4})"
        pattern_int = r"(?:(?<= )[-]*\d+|^[-]*\d+)"

        if len(re.findall(pattern_datetime, value)) == 1:
            value = value.replace('t', "T").replace('z', 'Z')
            return f'{value}^^http://www.w3.org/2001/XMLSchema#dateTime'
        elif len(re.findall(pattern_date, value)) == 1:
            if value.__contains__('-'):
                return f'{value}^^http://www.w3.org/2001/XMLSchema#date'
            elif value.__contains__('/'):
                fields = value.split('/')
                value = f"{fields[2]}-{fields[0]}-{fields[1]}"
                return f'{value}^^http://www.w3.org/2001/XMLSchema#date'
            else:
                if value.__contains__('jan.'):
                    month = '01'
                elif value.__contains__('feb.'):
                    month = '02'
                elif value.__contains__('mar.'):
                    month = '03'
                elif value.__contains__('apr.'):
                    month = '04'
                elif value.__contains__('may'):
                    month = '05'
                elif value.__contains__('jun.'):
                    month = '06'
                elif value.__contains__('jul.'):
                    month = '07'
                elif value.__contains__('aug.'):
                    month = '08'
                elif value.__contains__('sep.'):
                    month = '09'
                elif value.__contains__('oct.'):
                    month = '10'
                elif value.__contains__('nov.'):
                    month = '11'
                elif value.__contains__('dec.'):
                    month = '12'
                pattern = "(?<=the )\d+"
                day = re.findall(pattern, value)[0]
                if len(day) == 1:
                    day = f"0{day}"
                pattern = "(?<=, )\d+"
                year = re.findall(pattern, value)[0]
                return f'{year}-{month}-{day}^^http://www.w3.org/2001/XMLSchema#date'
        elif len(re.findall(pattern_yearmonth, value)) == 1:
            return f'{value}^^http://www.w3.org/2001/XMLSchema#gYearMonth'
        elif len(re.findall(pattern_float, value)) == 1:
            return f'{value}^^http://www.w3.org/2001/XMLSchema#float'
        elif len(re.findall(pattern_year, value)) == 1 and int(value) <= 2015:
            return f'{value}^^http://www.w3.org/2001/XMLSchema#gYear'
        else:
            return f'{value}^^http://www.w3.org/2001/XMLSchema#integer'
