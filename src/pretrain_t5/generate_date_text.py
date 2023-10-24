"""
List of Date Formats
"""

import random
import datetime

MONTH_DAYS = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

MONTHS = {
    1: ["January", "Jan", "01"],
    2: ["February", "Feb", "02"],
    3: ["March", "Mar", "03"],
    4: ["April", "Apr", "04"],
    5: ["May", "May", "05"],
    6: ["June", "Jun", "06"],
    7: ["July", "Jul", "07"],
    8: ["August", "Aug", "08"],
    9: ["September", "Sep", "09"],
    10: ["October", "Oct", "10"],
    11: ["November", "Nov", "11"],
    12: ["December", "Dec", "12"],
}

RULES = [
    "MM/DD/YY",
    "DD/MM/YY",
    "YY/MM/DD",
    "Month D, Yr",
    "M/D/YY",
    "D/M/YY",
    "YY/M/D",
    "bM/bD/YY",
    "YY/bM/bD",
    "MMDDYY",
    "DDMMYY",
    'YYMMDD',
    "MonDDYY",
    "DDMonYY",
    "YYMonDD",
    "D Month, Yr",
    "Yr, Month D",
    "Mon-DD-YYYY",
    "DD-Mon-YYYY",
    "YYYY-Mon-DD",
    "Mon DD, YYYY",
    "DD Mon, YYYY",
    "YYYY, Mon DD",
]
    # "day/YY",

YEAR_FORMATS=[
    "YYYY",
    "YY",
    "Yr",
]

MONTH_FORMATS = [
    "MM",
    "Month",
    "Mon",
    "bM",
    "M",
]

DAY_FORMATS = [
    "DD",
    "bD",
    "D",
]

def date_format_transform(date):
    date = str(date)
    return f"{date[:4]}-{date[4:6]}-{date[6:]}"


def covert_date_to_text(year, month=None, day=None):
    assert year is not None
    if not month:
        month = 0
    if not day:
        day = 0
    year, month, day = int(year), int(month), int(day)

    # assert month >= 1 and month <= 12 and day <= 31

    year_signal, month_signal, day_signal = False, False, False
    rule = random.choice(RULES)
    
    for year_fromat in YEAR_FORMATS:
        if year_fromat in rule:
            rule = rule.replace(year_fromat, str(year))
            year_signal = True
            break
    
    for day_format in DAY_FORMATS:
        if day_format in rule:
            if day_format == "DD":
                if day == 0 or month == 0:
                    rule = rule.replace(day_format, "")
                else:
                    day_str = '0'+str(day) if len(str(day)) == 1 else str(day)
                    rule = rule.replace(day_format, day_str)
            elif day_format == "bD":
                if day == 0 or month == 0:
                    rule = rule.replace(day_format, "")
                else:
                    day_str = ' '+str(day) if len(str(day)) == 1 else str(day)
                    rule = rule.replace(day_format, day_str)
            elif day_format == "D":
                if day == 0 or month == 0:
                    rule = rule.replace(day_format, "")
                else:
                    rule = rule.replace(day_format, str(day))
            day_signal = True
            break
    

    for month_format in MONTH_FORMATS:
        if month_format in rule:
            if month_format == "MM":
                if month == 0:
                    rule = rule.replace(month_format, "")
                else:
                    rule = rule.replace(month_format, str(MONTHS[month][2]))
            elif month_format == "Month":
                if month == 0:
                    rule = rule.replace(month_format, "")
                else:
                    rule = rule.replace(month_format, str(MONTHS[month][0]))
            elif month_format == "Mon":
                if month == 0:
                    rule = rule.replace(month_format, "")
                else:
                    rule = rule.replace(month_format, str(MONTHS[month][1]))
            elif month_format == "bM":
                if month == 0:
                    rule = rule.replace(month_format, "")
                else:
                    month_str = (" " + str(int(MONTHS[month][2])))[-2:]
                    rule = rule.replace(month_format, month_str)
            elif month_format == "M":
                if month == 0:
                    rule = rule.replace(month_format, "")
                else:
                    rule = rule.replace(month_format, str(int(MONTHS[month][2])))
            month_signal = True
            break
    
    if year_signal and month_signal and day_signal:
        return rule
    return None

def generate_random_date_inside_span(span_start, span_end):
    # start_date = datetime.date.fromisoformat(date_format_transform(span_start))
    # end_date = datetime.date.fromisoformat(date_format_transform(span_end))
    random_date = datetime.date.fromordinal(
        random.randint(span_start, span_end)
    )
    if random.random() < 0.5:
        input_month = None
        input_day = None
        year, month, day = random_date.isoformat().split("-")
        success = False
        next_day = random_date.day
        while not success:
            try:
                next_year = random_date.replace(year=int(year)+1, day=next_day).isoformat().split("-")[0]
                success = True
            except:
                next_day -= 1
                continue
        date_span = (f"{year}0101", f"{next_year}0101")
        date_text = covert_date_to_text(year, month=None, day=None)
    else:
        if random.random() < 0.5:
            input_day = None
            year, month, day = random_date.isoformat().split("-")
            if int(month) < 12:
                next_year = year
                next_month = random_date.replace(month=int(month)+1, day=min(MONTH_DAYS[int(month)+1], random_date.day)).isoformat().split("-")[1]
            elif int(month) == 12:
                #next_year = random_date.replace(year=int(year)+1).isoformat().split("-")[0]
                success = False
                assign_day = random_date.day
                while not success:
                    try:
                        next_year = random_date.replace(year=int(year)+1, day=assign_day).isoformat().split("-")[0]
                        success = True
                    except ValueError:
                        assign_day -= 1
                        assign_day = max(1, assign_day)
                        continue
                next_month = "01"
            date_span = (f"{year}{month}01", f"{next_year}{next_month}01")
            date_text = covert_date_to_text(year, month=month, day=None)
        else:
            year, month, day = random_date.isoformat().split("-")
            span_a = ''.join(random_date.isoformat().split("-"))
            span_b = ''.join(
                datetime.date.fromordinal(random_date.toordinal()+1).isoformat().split("-")
            )
            date_span = (span_a, span_b)
            date_text = covert_date_to_text(year, month=month, day=day)
    return date_text, date_span




if __name__ == "__main__":
    for i in range(20):
        print(
            covert_date_to_text(
                random.randint(1700, 2022),
                month=random.randint(1,12),
                day=random.randint(1,31),
            )
        )
