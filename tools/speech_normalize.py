#!/usr/bin/env python3
"""Text normalization for speech: numbers, symbols and abbreviations written out as words, before the sentence split
(validation/speech-normalize-v1, the "rules" arm). American English. Deterministic; nothing but the matched spans
changes. The style follows the experiment's guide: cardinals without "and"; years, decades and house numbers in pairs;
"point" decimals; "three dollars and fifty cents"; "seven forty-five", "seven oh five", "nine o'clock"; phone numbers
digit by digit ("oh" for zero); titles and abbreviations in full; "a.m."/"p.m." and acronyms left as written.

Matching is ASCII ((?a): digits, word characters, spaces and word boundaries are ASCII), so the native port (lfm/speech_normalize.hpp) can match it
byte for byte. Changes after validation/speech-normalize-v1 chose these rules (its held-out misses): "AM"/"PM" in
capitals count as a.m./p.m. after a time; "911" and a 5-digit number after "zip" or "postal" are read digit by digit;
a degree sign without C or F no longer takes the space after it ("20° today" read "twenty degreestoday"). The
normalizer audit (validation/audio-assurance-v1/normalizer-audit.md) added: ISO and day-first dates; international and
dotted phone numbers; "+" on and between numbers; feet and inches; dotted versions and IPv4 addresses; scientific
notation and powers; money signs, ranges, "k/M/B" scales and fractions of a cent; durations h:mm:ss; "3:30pm"; ratios;
percent ranges; "'90s"; mixed numbers, "50/50" and "24/7"; plurals of numbers ("747s"); dimensions ("4x4"); "-" and "/" in an equation; a
grouped whole part in a decimal ("1,234.5"); a year after a month name ("May 2020", "Dec. 31, 1999"); and a space
between a number and the letters written right after it ("22B", "6am"). Its red-team rounds narrowed these (each
rule reads as before the audit where it cannot be made right) and added: a subtraction read by the range rule when
"= <number>" follows; decimal ranges; "2^31-1"; a range before a mixed number or a time with a.m./p.m.; and
the real-text gate tools/normalizer_gate.py.

The rules, in order (each sees the previous one's output, and the words it writes hold no digits):
  phone numbers; money; numeric dates (M/D/YYYY); times; month abbreviations before a number; month-day dates;
  temperatures; percentages; addresses (house number + capitalized name + street suffix); decades; numbers with units;
  ordinals; fractions; ranges; "#N" and "No. N"; math symbols between spaces; negative numbers; decimals; integers
  (years by context, long or zero-led digit strings digit by digit, otherwise cardinals); titles and abbreviations.
An abbreviation or unit that ends a sentence (end of text, or a space and a capital follow) keeps its period; titles
and "vs." (a name follows them) never do.
Usage: speech_normalize.py "TEXT"   (prints the normalized text)
"""
import re
import sys

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)  # numbers of any length, as the native port

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (10 ** 3, "thousand")]
ORDINAL_WORDS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                 "nine": "ninth", "twelve": "twelfth"}
MONTHS = "January February March April May June July August September October November December".split()
MONTH_ABBR = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "Jun": "June", "Jul": "July",
              "Aug": "August", "Sep": "September", "Sept": "September", "Oct": "October", "Nov": "November",
              "Dec": "December"}
YEAR_CUES = {"in", "since", "by", "from", "until", "till", "of", "during", "year", "circa", "before", "after", "around",
             "early", "late", "mid", "spring", "summer", "fall", "autumn", "winter", "through", "to"}
STREET = {"St": "Street", "Street": "Street", "Ave": "Avenue", "Avenue": "Avenue", "Rd": "Road", "Road": "Road",
          "Blvd": "Boulevard", "Boulevard": "Boulevard", "Dr": "Drive", "Drive": "Drive", "Ln": "Lane", "Lane": "Lane",
          "Way": "Way", "Ct": "Court", "Court": "Court", "Pl": "Place", "Place": "Place"}
UNITS = {"km": ("kilometer", "kilometers"), "kg": ("kilogram", "kilograms"), "cm": ("centimeter", "centimeters"),
         "mm": ("millimeter", "millimeters"), "mg": ("milligram", "milligrams"), "ml": ("milliliter", "milliliters"),
         "mph": ("mile per hour", "miles per hour"), "kph": ("kilometer per hour", "kilometers per hour"),
         "ft": ("foot", "feet"), "lb": ("pound", "pounds"), "lbs": ("pound", "pounds"), "oz": ("ounce", "ounces"),
         "mi": ("mile", "miles"), "GB": ("gigabyte", "gigabytes"), "MB": ("megabyte", "megabytes"),
         "kWh": ("kilowatt hour", "kilowatt hours"), "hrs": ("hour", "hours"), "hr": ("hour", "hours"),
         "min": ("minute", "minutes"), "sec": ("second", "seconds")}
CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
            "€": ("euro", "euros", "cent", "cents")}
TITLES = {"vs": "versus", "Mr": "Mister", "Mrs": "Missus", "Ms": "Miz", "Dr": "Doctor", "Prof": "Professor", "Mt": "Mount",
          "Jr": "Junior", "Sr": "Senior", "Gov": "Governor", "Sen": "Senator", "Rep": "Representative",
          "Gen": "General", "Capt": "Captain", "Lt": "Lieutenant", "Sgt": "Sergeant"}
ABBREVIATIONS = {"etc": "et cetera", "e.g": "for example", "i.e": "that is", "approx": "approximately",
                 "Apt": "Apartment", "Dept": "Department", "Inc": "Incorporated", "Corp": "Corporation",
                 "Co": "Company", "Ave": "Avenue", "Blvd": "Boulevard", "Rd": "Road", "est": "established",
                 "min": "minutes", "max": "maximum", "govt": "government"}


def cardinal(n):
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + cardinal(n % 100) if n % 100 else "")
    if n >= 10 ** 12:  # "... trillion" groups of 12 digits, built in one pass (no recursion per group, as the native port)
        d = str(n)
        head = len(d) - 12 * ((len(d) - 1) // 12)
        words = cardinal(int(d[:head]))
        for k in range(head, len(d), 12):
            g = int(d[k:k + 12])
            words += " trillion" + (" " + cardinal(g) if g else "")
        return words
    for value, name in SCALES:
        if n >= value:
            return cardinal(n // value) + " " + name + (" " + cardinal(n % value) if n % value else "")
    raise ValueError(n)


def ordinal(n):
    words = cardinal(n)
    head, sep, last = words.rpartition("-" if "-" in words.split(" ")[-1] else " ")
    if last in ORDINAL_WORDS:
        last = ORDINAL_WORDS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def pairs(n):
    """Years and house numbers: 1984 nineteen eighty-four, 1905 nineteen oh five, 1900 nineteen hundred, 2000 two
    thousand, 2005 two thousand five, 2026 twenty twenty-six, 221 two twenty-one, 100 one hundred."""
    if n < 100 or n % 1000 == 0 and n < 10000 or 2000 <= n < 2010:
        return cardinal(n)
    if n < 1000:
        return cardinal(n) if n % 100 == 0 else ONES[n // 100] + " " + (cardinal(n % 100) if n % 100 >= 10 else
                                                                         "oh " + ONES[n % 100])
    hi, lo = divmod(n, 100)
    if lo == 0:
        return cardinal(hi) + " hundred"
    return cardinal(hi) + " " + (cardinal(lo) if lo >= 10 else "oh " + ONES[lo])


def plural_pairs(words):
    """'nineteen ninety' -> 'nineteen nineties' (decades)."""
    head, sep, last = words.rpartition(" ")
    last = last[:-1] + "ies" if last.endswith("y") else last + "s"
    return head + sep + last


def digits(s, zero="zero"):
    return " ".join(zero if c == "0" else ONES[int(c)] for c in s)


def number(s):
    """A plain number string (commas allowed, a decimal part allowed) in words."""
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        return cardinal(int(whole or "0")) + " point " + digits(frac)
    return cardinal(int(s))


def amount(value_words, one, many, is_one):
    return f"{value_words} {one if is_one else many}"


def power(n):
    """The exponent after "to the": 2 squared, 3 cubed, otherwise an ordinal ("to the sixth")."""
    return {2: "squared", 3: "cubed"}.get(n, "to the " + ordinal(n)) if n >= 0 else "to the minus " + ordinal(-n)


def exponent(sign, digits_):
    """An exponent as written: a whole one through power(), a decimal one as a number ("to the zero point five")."""
    neg = sign in ("-", "−")
    if "." in digits_:
        return "to the " + ("minus " if neg else "") + number(digits_)
    return power(-int(digits_) if neg else int(digits_))


def fraction_words(n, d):
    if d == 2:
        return cardinal(n) + (" half" if n == 1 else " halves")
    if d == 4:
        return cardinal(n) + (" quarter" if n == 1 else " quarters")
    return cardinal(n) + " " + ordinal(d) + ("" if n == 1 else "s")


LETTERS = re.compile(r"(?a)[A-Za-z]+")
NO_GLUE = {"s", "ers"}  # "747s" is a plural, "49ers" a name: the letters stay on the number, as before the audit


def decimal_less(a, b):
    """a < b for two decimal strings ("2", "3.5"), exactly."""
    (ai, _, af), (bi, _, bf) = a.partition("."), b.partition(".")
    ai, bi = ai.lstrip("0") or "0", bi.lstrip("0") or "0"
    if len(ai) != len(bi):
        return len(ai) < len(bi)
    if ai != bi:
        return ai < bi
    w = max(len(af), len(bf))
    return af.ljust(w, "0") < bf.ljust(w, "0")


def glue(t, end, whole=True):
    """A space when letters follow a number directly ("22B", "3D", "6am"), so the words never run into them; after a
    whole number, not before a plural "s" or "ers" ("747s"; after a decimal, "2.5s" reads "two point five s")."""
    m = LETTERS.match(t, end)
    return " " if m and not (whole and m.group(0) in NO_GLUE) else ""


def plural_words(words):
    """'six' -> 'sixes', 'seven hundred forty-seven' -> 'seven hundred forty-sevens', 'twenty' -> 'twenties'."""
    if words.endswith("y"):
        return words[:-1] + "ies"
    return words + ("es" if words.endswith("x") else "s")


LINK = re.compile(r"(?a)@|://|www\.|[A-Za-z0-9]\.(?:com|org|net|edu|gov|io|dev|ai|app|co|uk|de)(?![A-Za-z0-9])")


def links(t):
    """The spans of whitespace-delimited tokens that are links or addresses (an "@", "://", "www." or a domain): the
    audit's rules leave them as written."""
    return [m.span() for m in re.finditer(r"(?a)\S+", t) if LINK.search(m.group(0))]


def in_links(spans, i):
    import bisect
    k = bisect.bisect_right(spans, (i, float("inf"))) - 1
    return k >= 0 and spans[k][0] <= i < spans[k][1]


def sub(pattern, repl, t):
    """re.sub for the audit's rules: a match that starts inside a link is left as written."""
    spans = links(t)

    def f(m):
        if spans and in_links(spans, m.start()):
            return m.group(0)
        return repl(m) if callable(repl) else m.expand(repl)
    return re.sub(pattern, f, t)


def word_before(t, i):
    """The word right before position i (letters, an optional colon, then one or more spaces), lowercased, within 40
    characters; or ""."""
    m = re.search(r"(?a)([A-Za-z]+):? +\Z", t[max(0, i - 40):i])
    return m.group(1).lower() if m else ""


def words_near(t, start, end):
    """The last three words in the 30 characters before start, and the word after end (spaces, then letters)."""
    before = re.findall(r"(?a)[A-Za-z]+", t[max(0, start - 30):start])[-3:]
    after = re.compile(r"(?a) +([A-Za-z]+)").match(t, end)
    return [w.lower() for w in before] + ([after.group(1).lower()] if after else [])


DURATION_CUES = {"in", "ran", "run", "time", "times", "lasted", "lasts", "took", "takes", "finished", "clocked",
                 "record", "split", "length", "runtime", "duration", "runs"}
CLOCK_CUES = {"at", "by", "until", "till", "from", "since"}
RATIO_CUES = {"ratio", "ratios", "aspect", "odds", "scale", "mix", "mixed", "mixture", "proportion", "proportions"}
VERSION_CUES = {"version", "versions", "release", "releases", "v", "build", "update", "upgrade", "patch"}
IP_CUES = {"ip", "address", "ping", "connect", "server", "host", "mask", "netmask", "gateway", "dns", "subnet", "router"}


MUSIC_CUES = {"time", "signature", "rhythm", "meter", "metre", "beat", "beats", "measure", "groove", "tempo"}
OPS = {"minus", "plus", "equals", "times"}
OP_AFTER = re.compile(r"(?a) +([a-z]+)")
SCORE_CUES = {"rated", "rating", "ratings", "score", "scored", "scores", "star", "stars", "grade", "graded", "review",
              "reviews", "reviewed"}
ARITHMETIC = set("=*×+")
WS = " \t\n\r\f\v"


def arithmetic_near(t, start, end):
    """The nearest non-space character before start or after end is one of = * × +, or a parenthesis with one of them
    beyond it ("(3/3)*4"): a fraction in arithmetic."""
    def back(i):
        while i > 0 and t[i - 1] in WS:
            i -= 1
        return i

    def fwd(j):
        while j < len(t) and t[j] in WS:
            j += 1
        return j
    i, j = back(start), fwd(end)
    if i > 0 and t[i - 1] == "(":
        i = back(i - 1)
    if j < len(t) and t[j] == ")":
        j = fwd(j + 1)
    return (i > 0 and t[i - 1] in ARITHMETIC) or (j < len(t) and t[j] in ARITHMETIC)


def operand_before(before):
    """before (right-stripped) ends with + * ×, or with "=" after a digit or a one-letter variable ("x=", not
    "established=")."""
    if before.endswith(("+", "*", "×")):
        return True
    if not before.endswith("="):
        return False
    p = before[:-1].rstrip()
    if not p or not p[-1].isascii():
        return False
    if p[-1].isdigit():
        return True
    return p[-1].isalpha() and (len(p) == 1 or not (p[-2].isascii() and p[-2].isalnum()))


def angle_before(t, i):
    r"""(?:°|\bdeg\.?) *\Z before i: the minutes and seconds of an angle follow."""
    q = i
    while q > 0 and t[q - 1] == " ":
        q -= 1

    def deg(k):
        return k >= 3 and t[k - 3:k] == "deg" and (k == 3 or not re.match(r"(?a)\w", t[k - 4]))
    return (q > 0 and t[q - 1] == "°") or deg(q) or (q > 0 and t[q - 1] == "." and deg(q - 1))


def normalize(text):
    t = text
    # ISO dates: 2024-03-15 (audio-assurance-v1 normalizer audit, as the rules below that use sub())
    t = sub(r"(?a)\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b(?!-)",
            lambda m: f"{MONTHS[int(m.group(2)) - 1]} {ordinal(int(m.group(3)))}, {pairs(int(m.group(1)))}", t)

    # international numbers: +1 415 555 0123, +44 20 7946 0958 (8 digits or more; otherwise left to the "+" rule)
    def international(m):
        groups = [m.group(1)] + re.findall(r"(?a)\d+", m.group(2))
        if sum(map(len, groups)) < 8:
            return m.group(0)
        return "plus " + ", ".join(digits(g, "oh") for g in groups)
    t = sub(r"(?a)(?<![\w.+])\+(\d{1,3})((?:[ .-]\d{1,4}(?!\d)){2,5})\b(?![.-]?\d)", international, t)
    # phone numbers: 555-0142, (555) 123-4567, 555-123-4567, 555.123.4567
    t = re.sub(r"(?a)\(?\b(\d{3})\)?[-. ](\d{3})[-.](\d{4})\b",
               lambda m: ", ".join(digits(g, "oh") for g in m.groups()), t)
    t = re.sub(r"(?a)(?<![\d.,-])\b(\d{3})-(\d{4})\b(?![-\d])", lambda m: ", ".join(digits(g, "oh") for g in m.groups()), t)
    # a plus sign on a number ("+5%", "+$3") or between digits ("2+2")
    t = sub(r"(?a)(?<![\w.+])\+(?=[$£€]?\d)", "plus ", t)
    t = sub(r"(?a)(?<=\d)\+(?=\d)", " plus ", t)

    # feet and inches: 5'10", 6' 2" (under 12 inches; not the minutes and seconds of an angle: 34° 6' 3", 5 deg. 49' 30")
    def feet(m):
        if int(m.group(2)) > 11 or angle_before(t, m.start()):
            return m.group(0)
        return (f"{cardinal(int(m.group(1)))} {'foot' if m.group(1) == '1' else 'feet'} "
                f"{cardinal(int(m.group(2)))} {'inch' if m.group(2) == '1' else 'inches'}")
    t = sub(r"(?a)(?<![\w.,'])(\d+)' ?(\d{1,2})(?:\"|”|'')", feet, t)

    # dotted numbers of three groups or more: a date (15.03.2024 day first; 3.15.2024 month first when the day is above
    # 12; 2024.03.15); four valid octets: an IPv4 address digit by digit with "dot" (a three-digit group, or a network
    # word before it), else a version; grouped thousands (1.234.567, no version word before it); otherwise a version
    def dotted(m):
        groups = m.group(0).split(".")
        near = set(words_near(t, m.start(), m.start())[:3])
        if len(groups) == 3 and len(groups[2]) == 4 and 1000 <= int(groups[2]) <= 2099 and len(groups[0]) <= 2 \
                and len(groups[1]) <= 2:
            a, b = int(groups[0]), int(groups[1])
            if 1 <= a <= 31 and 1 <= b <= 12:
                return f"{MONTHS[b - 1]} {ordinal(a)}, {pairs(int(groups[2]))}"
            if 1 <= a <= 12 and 13 <= b <= 31:
                return f"{MONTHS[a - 1]} {ordinal(b)}, {pairs(int(groups[2]))}"
        if len(groups) == 3 and len(groups[0]) == 4 and 1000 <= int(groups[0]) <= 2099 and len(groups[1]) == 2 \
                and len(groups[2]) == 2 and 1 <= int(groups[1]) <= 12 and 1 <= int(groups[2]) <= 31:
            return f"{MONTHS[int(groups[1]) - 1]} {ordinal(int(groups[2]))}, {pairs(int(groups[0]))}"
        octets = len(groups) == 4 and all(len(g) <= 3 and int(g) <= 255 for g in groups)
        if octets and (any(len(g) == 3 for g in groups) or near & IP_CUES):
            return " dot ".join(digits(g) for g in groups)
        if not octets and len(groups[0]) <= 3 and groups[0][0] != "0" and all(len(g) == 3 for g in groups[1:]) \
                and not near & (VERSION_CUES | IP_CUES):
            return cardinal(int("".join(groups)))
        return " point ".join(digits(g) if len(g) > 1 and g[0] == "0" else cardinal(int(g)) for g in groups)  # 22.04.3
    t = sub(r"(?a)(?<![\w.,/])\d+(?:\.\d+){2,}\b(?!\.\d)", dotted, t)

    # scientific notation and powers: 3e8, 6.02e23, 2E-5; 10^6, 2^10, 2^0.5 (a decimal exponent read as a number). Not
    # inside a token ("123e4567-e89b", "#3e5", "2E5-100"); a capital E without a decimal point or a sign ("Room 2E1") is
    # left as written
    def scientific(m):
        if m.group(2) == "E" and "." not in m.group(1) and not m.group(3):
            return m.group(0)
        return f"{number(m.group(1))} times ten {exponent(m.group(3), m.group(4))}"
    t = sub(r"(?a)(?<![\w.])[-−](?=\d+(?:\.\d+)?(?:\^|[eE][-+−]?\d))", "minus ", t)  # -10^9, -1e-5
    t = sub(r"(?a)(?<![\w.,/#-])(\d+(?:\.\d+)?)([eE])([-+−]?)(\d+(?:\.\d+)?)\b(?![-.]?\w)", scientific, t)
    t = sub(r"(?a)(?<![\w.,/-])(\d+(?:\.\d+)?)\^([-−]?)(\d+(?:\.\d+)?)\b(?!\.\d)(?:( ?)[-−–]\4(?=\d))?",
            lambda m: f"{number(m.group(1))} {exponent(m.group(2), m.group(3))}" + (" minus " if m.group(4) is not None else ""),
            t)  # 2^31-1: "two to the thirty-first minus one"
    # a minus sign on money ("-$3 million"); money ranges ("$10-$20", "$10 - 20": no space or a space on both sides),
    # but not a subtraction ("$5 - $3 = $2")
    t = sub(r"(?a)(?<![\w.])[-−](?=[$£€]\d)", "minus ", t)
    second = re.compile(r"(?a)[$£€]?(\d[\d,]*(?:\.\d+)?)")
    chain = re.compile(r"(?a)\s*[-–]\s*[$£€]?\d")

    def money_range(m):  # ascending, not chained ("$60-$13-$6"), no "=" later in the clause ("$450 - $75 - $120 = $255")
        b = second.match(t, m.end())
        i = m.start()
        while i > 0 and t[i - 1] in " \t\n\r\f\v":
            i -= 1
        if i > 0 and t[i - 1] in "-–":  # the end of a chain: an amount, then a dash, before it (not a bullet "- $5 - $10")
            k = i - 1
            while k > 0 and t[k - 1] in WS:
                k -= 1
            if k > 0 and t[k - 1].isascii() and t[k - 1].isdigit():
                return m.group(0)
        if not decimal_less(m.group(2).replace(",", ""), b.group(1).replace(",", "")) or chain.match(t, b.end()):
            return m.group(0)
        if "=" in re.split(r"(?a)[;!?\n]|\.(?:\s|$)", t[b.end():b.end() + 120], maxsplit=1)[0]:
            return m.group(0)
        return m.group(1) + " to "
    t = sub(r"(?a)(?<![\w.])([$£€](\d[\d,]*(?:\.\d+)?))( ?)[-–]\3(?=[$£€]?\d)", money_range, t)

    # money: $3.50, $129, $1.2 million, £5, €20.05
    def money(m):
        one, many, sub_one, sub_many = CURRENCY[m.group(1)]
        value = m.group(2).replace(",", "")
        scale = m.group(4) or {"k": "thousand", "K": "thousand", "M": "million", "B": "billion", "bn": "billion",
                               None: None}[m.group(5)]
        if scale:
            return f"{number(value)} {scale} {many}"
        whole, _, frac = value.partition(".")
        if len(frac) > 2:  # $3.999: not a whole number of cents
            return f"{number(value)} {many}"
        cents = int((frac + "00")[:2]) if frac else 0
        words = amount(cardinal(int(whole)), one, many, int(whole) == 1) if int(whole) or not cents else ""
        if cents:
            words += (" and " if words else "") + amount(cardinal(cents), sub_one, sub_many, cents == 1)
        return words
    t = re.sub(r"(?a)([$£€])(\d[\d,]*(?:\.\d+)?)(\s+(million|billion|trillion|thousand)\b|(bn|[kKMB])\b)?", money, t)
    # numeric dates (US month/day/year)
    t = re.sub(r"(?a)\b(1[0-2]|0?[1-9])/(3[01]|[12]\d|0?[1-9])/(\d{4})\b",
               lambda m: f"{MONTHS[int(m.group(1)) - 1]} {ordinal(int(m.group(2)))}, {pairs(int(m.group(3)))}", t)
    # day first when the day cannot be a month: 15/03/1990
    t = sub(r"(?a)\b(1[3-9]|2\d|3[01])/(1[0-2]|0?[1-9])/(\d{4})\b",
            lambda m: f"{MONTHS[int(m.group(2)) - 1]} {ordinal(int(m.group(1)))}, {pairs(int(m.group(3)))}", t)

    # h:mm:ss(.s): a duration ("in 1:35:10", "ran", "runs", "duration", or more than 23 hours) in hours, minutes and
    # seconds, a duration word beating a leading zero ("runs 00:03:25"); a time of day (a.m./p.m. after it, a leading
    # zero, "at"/"by"/..., or 13 hours or more) as a clock; otherwise as before the audit
    meridiem_after = re.compile(r"(?a)\s*[aApP]\.?[mM]\b")

    def hms(m):
        h, mm, sec = int(m.group(1)), int(m.group(2)), m.group(3)
        sec_zero, sec_one = not sec.replace("0", "").replace(".", ""), sec == "01"
        sec_words = f"{number(sec)} second{'' if sec_one else 's'}"
        w = word_before(t, m.start())
        meridiem = bool(meridiem_after.match(t, m.end()))
        if h > 23 or w in DURATION_CUES and not meridiem:
            words = [f"{cardinal(v)} {unit}{'' if v == 1 else 's'}" for v, unit in ((h, "hour"), (mm, "minute")) if v]
            words += [] if sec_zero else [sec_words]
            return " ".join(words) if words else "zero seconds"
        if meridiem or len(m.group(1)) == 2 and m.group(1)[0] == "0" or w in CLOCK_CUES or h >= 13:
            if mm == 0:
                words = cardinal(h) + ("" if meridiem else " o'clock")
            else:
                words = cardinal(h) + " " + (cardinal(mm) if mm >= 10 else "oh " + ONES[mm])
            return words + ("" if sec_zero else " and " + sec_words)
        return m.group(0)
    t = sub(r"(?a)(?<!:)\b(\d{1,2}):([0-5]\d):([0-5]\d(?:\.\d+)?)\b(?![:.]\d|:)", hms, t)

    # a range before a time with a.m./p.m.: "8-9:30am" -> "8 to 9:30am"
    t = sub(r"(?a)(?<![\w.,:-])(\d{1,2})( ?)[-–]\2(?=\d{1,2}(?::[0-5]\d)?\s*[aApP]\.?[mM]\b)", r"\1 to ", t)
    # times: 7:45, 9:00, 10:30 (o'clock only when no a.m./p.m. follows); a lap time's fraction: 1:23.456 (the audit)
    def clock(m):
        h, mm = int(m.group(1)), int(m.group(2))
        if m.group(3):
            return (cardinal(h) + " " + (cardinal(mm) if mm >= 10 else digits(m.group(2), "oh")) + " point " +
                    digits(m.group(3)[1:]) + glue(t, m.end(), whole=False))
        if mm == 0:
            return cardinal(h) + ("" if m.group(4) else " o'clock") + glue(t, m.end())
        return cardinal(h) + " " + (cardinal(mm) if mm >= 10 else "oh " + ONES[mm]) + glue(t, m.end())
    t = re.sub(r"(?a)\b([01]?\d|2[0-3]):([0-5]\d)(\.\d+)?(?:\b|(?=[aApP]\.?[mM]\b))(?=(\s*[aApP]\.?[mM]\b\.?)?)", clock, t)

    # a time with a dot before a.m./p.m.: 7.30pm, 12.35 am
    t = sub(r"(?a)\b(1[0-2]|0?[1-9])\.([0-5]\d)(?=\s*[aApP]\.?[mM]\b)",
            lambda m: cardinal(int(m.group(1))) + ("" if m.group(2) == "00" else " " + (
                cardinal(int(m.group(2))) if int(m.group(2)) >= 10 else "oh " + ONES[int(m.group(2))])) + glue(t, m.end()), t)

    # ratios: 3:1, 16:9, 1:100 (what the times above leave), only with a ratio word near ("ratio", "aspect", "odds",
    # "scale", "mix"): race times, verses and ports stay as before the audit
    def ratio(m):
        if not set(words_near(t, m.start(), m.end())) & RATIO_CUES:
            return m.group(0)
        return f"{cardinal(int(m.group(1)))} to {number(m.group(2))}"
    t = sub(r"(?a)(?<![\w.,:])(\d+):((?:[1-9]\d*|0)(?:\.\d+)?)\b(?![:,.]\d|:)", ratio, t)
    # month abbreviations before a number; month-day dates
    t = re.sub(r"(?a)\b(" + "|".join(MONTH_ABBR) + r")\.(?=\s+\d)", lambda m: MONTH_ABBR[m.group(1)], t)
    t = re.sub(r"(?a)\b(" + "|".join(MONTHS) + r")\s+(3[01]|[12]\d|0?[1-9])(st|nd|rd|th)?\b(?![\d:])",
               lambda m: f"{m.group(1)} {ordinal(int(m.group(2)))}", t)
    # temperatures: -5°C, 41°F, 20°
    t = re.sub(r"(?a)(?<![\w.])([-−])?(\d+(?:\.\d+)?)\s*°(?:\s*([CF])\b)?",
               lambda m: ("minus " if m.group(1) else "") + number(m.group(2)) + " degree" +
               ("" if m.group(2) == "1" else "s") + {"C": " Celsius", "F": " Fahrenheit", None: ""}[m.group(3)], t)
    # percentages; a range first ("2-3%", "2 - 3%"; ascending, so not "100 -3%" or "fell 30 - 2%")
    pct_end = re.compile(r"(?a)\d+(?:\.\d+)?")
    t = sub(r"(?a)(?<![\w.,-])(\d+(?:\.\d+)?)( ?)[-–]\2(?=\d+(?:\.\d+)?\s*%)",
            lambda m: m.group(1) + " to " if decimal_less(m.group(1), pct_end.match(t, m.end()).group(0)) else m.group(0), t)
    t = re.sub(r"(?a)(?<![\w.])([-−])?(\d[\d,]*(?:\.\d+)?)\s*%",
               lambda m: ("minus " if m.group(1) else "") + number(m.group(2)) + " percent", t)
    # addresses: 1600 Main St., 221 Baker Street
    t = re.sub(r"(?a)\b(\d{1,5}) ((?:[A-Z][a-z]+ )+)(" + "|".join(STREET) + r")\b(\.)?",
               lambda m: f"{pairs(int(m.group(1)))} {m.group(2)}{STREET[m.group(3)]}" +
               ("." if m.group(4) and re.match(r"(?a)\s*$|\s+[A-Z]", t[m.end():]) else ""), t)
    # decades: 1990s, '90s, 80s
    t = re.sub(r"(?a)\b(1\d|20)(\d0)s\b", lambda m: plural_pairs(pairs(int(m.group(1) + m.group(2)))), t)
    t = sub(r"(?a)(?<![\w.])'?([1-9]0)s\b", lambda m: plural_pairs(cardinal(int(m.group(1)))), t)

    # numbers with units: 12 km, 2,800 ft, 0.5 mg
    def unit(m):  # a unit's own period goes, unless it also ends the sentence
        one, many = UNITS[m.group(2)]
        end = "." if m.group(3) and re.match(r"(?a)\s*$|\s+[A-Z\"'“(]", t[m.end():]) else ""
        return number(m.group(1)) + " " + (one if m.group(1) == "1" else many) + end
    t = re.sub(r"(?a)(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(" + "|".join(sorted(UNITS, key=len, reverse=True)) + r")\b(\.)?",
               unit, t)
    # ordinals: 1st, 22nd, 3rd
    t = re.sub(r"(?a)\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), t)

    # mixed numbers: 2 1/2, 1 3/4 (a whole number of one or two digits, a proper fraction in halves, thirds, quarters,
    # eighths or sixteenths: not "the 300 2/3 voted", "Week 3 4/5"); a range to one first: "1-1 1/2" -> "1 to 1 1/2"
    def mixed(m):
        w, n, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not 0 < n < d or d not in (2, 3, 4, 8, 16):
            return m.group(0)
        if n == 1:
            part = {2: "half", 4: "quarter"}.get(d) or ordinal(d)
            part = ("an " if part[0] in "aeiou" else "a ") + part
        else:
            part = fraction_words(n, d)
        return f"{cardinal(w)} and {part}"
    t = sub(r"(?a)(?<![\w.,])(\d{1,2})( ?)[-–]\2(?=\d{1,2} \d+/(?:2|3|4|8|16)\b(?!/))", r"\1 to ", t)  # 1-1 1/2
    t = sub(r"(?a)(?<![\w.,\-–−])(\d{1,2}) (\d+)/(\d+)\b(?!/)", mixed, t)

    # fractions: 1/2, 3/4, 2/3; 24/7, 50/50, 20/20 vision and 4/4 time as said; n/n "out of" after a score word; n/1 as
    # odds after "odds"; in a link or in arithmetic (= * × + ( ) next to it) always as before the audit
    spans = links(t)

    def fraction(m):
        n, d = int(m.group(1)), int(m.group(2))
        if spans and in_links(spans, m.start()) or arithmetic_near(t, m.start(), m.end()):
            return fraction_words(n, d)  # in arithmetic ("Gap size = 2/2 = 1", "60/60 = 1"): as before the audit
        near = set(words_near(t, m.start(), m.end()))
        if (n, d) in ((24, 7), (50, 50)) or n == d == 20 and "vision" in near or n == d == 4 and near & MUSIC_CUES:
            return cardinal(n) + " " + cardinal(d)  # 24/7, 50/50, 20/20 vision, 4/4 time
        if n == d and d > 1 and near & SCORE_CUES:
            return f"{cardinal(n)} out of {cardinal(d)}"  # "rated 10/10", "5/5 stars"
        if d == 1 and n > 1 and "odds" in near:
            return cardinal(n) + " to one"  # "odds of 9/1" (not "Starting 9/1": a date)
        return fraction_words(n, d)
    t = re.sub(r"(?a)(?<![\w/.])(\d+)/(\d+)\b(?!/)", fraction, t)

    # dimensions written without spaces: 4x4, 1920x1080, 2,560×1,440, 4x4x4, 4x4s (not hex: 0x10)
    def dimensions(m):
        parts = [number(g) for g in m.groups()[:3] if g]
        if m.group(4):
            parts[-1] = plural_words(parts[-1])
        return " by ".join(parts) + glue(t, m.end(), whole=False)  # "6.3cm" -> "six point three cm"
    dim = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    t = sub(r"(?a)(?<![\w.,*=+])(?!0x)" + dim + "[x×]" + dim + "(?:[x×]" + dim + ")?(?:(s)\\b|\\b|(?=[A-Za-z]))(?!,\\d)(?![*=+(])",
            dimensions, t)  # not in arithmetic ("24×7*3", "3x10(8)")

    # ranges: 25-30, 1990-1995
    def span(m):
        x, y = m.group(1), m.group(4)
        if m.group(5) is not None:  # "20 - 1 = 19", "3-1=2": a subtraction, "=" read with it (the audit)
            return f"{number(x)} minus {number(y)} equals "
        if "." in x + y:  # a decimal end ("1-1.25", "CI 1.2-3.4"): ascending, the same spacing on both sides, no year end
            year = any("." not in v and len(v) == 4 and 1000 <= int(v) <= 2099 for v in (x, y))
            if bool(m.group(2)) != bool(m.group(3)) or year or not decimal_less(x, y):
                return m.group(0)  # "1.5 -5", "610.12-1990", "5.0-0": as before the audit
            return f"{number(x)} to {number(y)}"
        a, b = int(x), int(y)
        both_years = all(1000 <= v <= 2099 for v in (a, b))
        return (pairs(a) if both_years else cardinal(a)) + " to " + (pairs(b) if both_years else cardinal(b))
    t = re.sub(r"(?a)(?<![\w.,-])(\d+(?:\.\d+)?)(\s?)[-–](\s?)(\d+(?:\.\d+)?)\b(?![-\d])(?!\.\d)(?:(\s*=\s*)(?=[-−]?\d))?", span, t)
    # "#12", "No. 14"
    t = re.sub(r"(?a)#\s?(\d+)\b", lambda m: "number " + cardinal(int(m.group(1))), t)
    t = re.sub(r"(?a)\bNo\.\s?(\d+)\b", lambda m: "number " + cardinal(int(m.group(1))), t)
    # in an equation ("10 - 3 = 7", "3-1=2", "8 / 2 = 4"): not a range, not a season ("2012 / 13"); "=" between digits
    # in an equation ("10 - 3 = 7" past a range, "3-1=2", "8 / 2 = 4"): not a range, not a season ("2012 / 13")
    t = sub(r"(?a)(?<=\d)( ?)[-−–]\1(?=\d[\d,.]*\s*=\s*[-−]?\d)", " minus ", t)
    t = sub(r"(?a)(?<=\d)( ?)/\1(?=\d[\d,.]*\s*=\s*[-−]?\d)", " divided by ", t)
    t = sub(r"(?a)(?<=\d)=(?=[-−]?\d)", " equals ", t)
    # math symbols with spaces around them; & anywhere between words
    t = re.sub(r"(?a)(?<=\s)=(?=\s)", "equals", t)
    t = re.sub(r"(?a)(?<=\s)\+(?=\s)", "plus", t)
    t = re.sub(r"(?a)(?<=\s)×(?=\s)", "times", t)
    t = re.sub(r"(?a)(?<=\s)&(?=\s)", "and", t)
    # negative numbers
    t = re.sub(r"(?a)(?<![\w.])[-−](?=\d)", "minus ", t)
    # decimals
    t = re.sub(r"(?a)(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)\.(\d+)(?![\d_])", lambda m: number(m.group(0)) + glue(t, m.end(), whole=False), t)

    # plurals of numbers: 747s, 6s, under-21s, 100s (not zero-led or long digit strings, which stay as before)
    def plural(m):
        s = m.group(1)
        if len(s) > 1 and s[0] == "0" or len(s) >= 7 or s == "911":
            return m.group(0)
        n = int(s)
        words = cardinal(n)
        if n in (100, 1000, 10 ** 6) and word_before(t, m.start()) != "the" and t[m.start() - 1:m.start()] != "-":
            words = words[4:]  # "100s of millions": "hundreds"; "the 100s" (a class, a century): "the one hundreds"
        return plural_words(words)
    t = sub(r"(?a)(?<![\w.,'])(\d+)s\b", plural, t)

    # integers: years by context, digit strings, cardinals
    def integer(m):
        s = m.group(0)
        if "," in s:
            return cardinal(int(s.replace(",", "")))
        if len(s) > 1 and s[0] == "0":
            return digits(s)
        if len(s) >= 7:  # a long digit string digit by digit, but a number in arithmetic ("2000000 minus 400000") whole
            after = OP_AFTER.match(t, m.end())
            if word_before(t, m.start()) == "equals" or after and after.group(1) in OPS:
                return cardinal(int(s))
            return digits(s)
        n = int(s)
        before_words = re.findall(r"(?a)[A-Za-z]+", t[:m.start()][-30:])
        if s == "911" or len(s) == 5 and any(w.lower() in ("zip", "postal") for w in before_words[-3:]):
            return digits(s)  # an emergency number; a ZIP code ("zip code is 12345")
        if len(s) == 4 and 1000 <= n <= 2099:
            before = t[:m.start()].rstrip()
            prev = re.findall(r"(?a)[A-Za-z]+", before[-20:])
            after = t[m.end():m.end() + 2]
            bare = re.search(r"(?a)(" + "|".join(MONTHS) + r"),?$", before)  # "May 2020", not "In March 1234 people"
            if operand_before(before):  # "3.5x=1050.", ".20*1800": an operand, not a year
                return cardinal(n)
            if (prev and prev[-1].lower() in YEAR_CUES) or re.search(r"(?a)(" + "|".join(MONTHS) + r") [\w-]+,?$", before) \
                    or bare and not re.match(r"(?a) +[a-z]", t[m.end():m.end() + 64]) \
                    or re.match(r"(?a)\s*[.,;:!?]|\s*$", after) and not re.search(r"(?a)\d", before[-3:]):
                return pairs(n)
        return cardinal(n)
    t = re.sub(r"(?a)(?<![\w.])\d{1,3}(?:,\d{3})+(?![\d,]*\d)|(?<![\w.,])\d+(?![\d,]*\d)",
               lambda m: integer(m) + glue(t, m.end()), t)

    # titles and abbreviations
    def abbreviation(m):
        word = m.group(1)
        follow = t[m.end():]
        if word == "St":  # Saint before a name, Street otherwise
            return "Saint" if re.match(r"(?a)\s+[A-Z]", follow) else "Street" + period(follow)
        if word == "Dr" and not re.match(r"(?a)\s+[A-Z]", follow):  # a name follows Doctor; a street name precedes Drive
            return "Drive" + period(follow)
        if word in TITLES:
            return TITLES[word]
        return ABBREVIATIONS[word] + period(follow)

    def period(follow):
        return "." if re.match(r"(?a)\s*$|\s+[A-Z\"'“(]", follow) else ""
    keys = sorted(set(TITLES) | set(ABBREVIATIONS) | {"St"}, key=len, reverse=True)
    t = re.sub(r"(?a)(?<![\w.])(" + "|".join(re.escape(k) for k in keys) + r")\.(?=[\s,;:!?)]|$)", abbreviation, t)
    t = re.sub(r"(?a)(?<![\w.])vs(?=\s)", "versus", t)
    return re.sub(r"(?a)[ \t]{2,}", " ", t)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(normalize(sys.argv[1]))
