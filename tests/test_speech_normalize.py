"""The native speech normalizer (native/include/lfm/speech_normalize.hpp) against its reference
(tools/speech_normalize.py), byte for byte, through native/tests/speech_normalize_tool.cpp built plain and with
ASan+UBSan.

Inputs: the experiment's 31 sentences (validation/speech-normalize-v1); every line of the local wikitext-2 and PG-19
test data that holds a digit, a symbol the rules read, or a period after a short word (when the data is present); and
30,000 generated strings mixing the rules' ingredients (numbers in every format they read, currency, degree, percent
and minus signs, dashes, times, dates, units, abbreviations, capitals, odd whitespace and punctuation, non-ASCII text).
Also: the rules' own expectations on a few sentences, and time on a long text (linear passes).
"""
import importlib.util
import json
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("speech_normalize", ROOT / "tools/speech_normalize.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)
VARIANTS = {"plain": ["-O2"], "asan": ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]}


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def tool(tmp_path_factory, request):
    if not shutil.which("g++"):
        pytest.skip("g++ required")
    binary = tmp_path_factory.mktemp("normalize") / f"speech-normalize-tool-{request.param}"
    subprocess.run(["g++", "-std=c++20", "-Wall", "-Wextra", "-Werror", *VARIANTS[request.param], "-I",
                    str(ROOT / "native/include"), str(ROOT / "native/tests/speech_normalize_tool.cpp"), "-o", str(binary)],
                   check=True)
    return binary


def native(tool, tmp_path, texts):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text("".join(json.dumps(t) + "\n" for t in texts), encoding="utf-8")
    subprocess.run([str(tool), str(src), str(dst)], check=True, capture_output=True, timeout=1800)
    return [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]


def compare(tool, tmp_path, texts):
    got = native(tool, tmp_path, texts)
    assert len(got) == len(texts)
    bad = [(t, g, reference.normalize(t)) for t, g in zip(texts, got) if g != reference.normalize(t)]
    assert not bad, "\n".join(f"in:  {t!r}\nC++: {g!r}\nref: {w!r}" for t, g, w in bad[:5]) + f"\n({len(bad)} differ)"


def test_experiment_sentences(tool, tmp_path):
    texts = [c["text"] for s in ("dev", "heldout")
             for c in json.load(open(ROOT / f"validation/speech-normalize-v1/cases-{s}.json"))["cases"]]
    compare(tool, tmp_path, texts)


def corpus_lines():
    data = ROOT.parent / "data"
    files = sorted((data / "wikitext-2-raw-v1").rglob("*")) + sorted((data / "pg19-test").rglob("*.txt"))[:20]
    want = re.compile(r"\d|[$£€%°#&+=×−–]|\b[A-Za-z]{1,6}\.")
    lines = []
    for f in files:
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if want.search(line) and len(line) < 4000:
                lines.append(line)
    return lines


def test_real_text(tool, tmp_path):
    lines = corpus_lines()
    if len(lines) < 1000:
        pytest.skip("corpus data not present")
    compare(tool, tmp_path, lines[:40000])


PIECES = [
    "", " ", "  ", "\t", "\n", ",", ".", ". ", ", ", ";", ":", "!", "?", "(", ")", "\"", "'", "“", "”", "-", "--", "–",
    "—", "/", "#", "%", "°", "°C", "°F", " °", "$", "£", "€", "−", "×", "&", "+", "=", "_", "x", "a", "A", "é", "日本",
    "0", "1", "7", "9", "12", "07", "389", "911", "1,500", "1,234,567", "12,34", "1,0000", "3.5", "0.25", "2.", ".5",
    "1984", "2026", "1500", "2000", "2005", "1905", "12345", "02139", "1234567", "99999999999999999999",
    "7:45", "9:00", "23:59", "24:00", "1:5", "12:30:45", "AM", "PM", "a.m.", "p.m.", "am", "pm",
    "1st", "2nd", "3rd", "11th", "22nd", "1/2", "3/4", "2/3", "11/9/2026", "13/45/2020", "25-30", "1990-1995",
    "555-0142", "(555) 123-4567", "555-123-4567", "km", "kg", "mph", "ft", "ft.", "lbs", "min", "hr", "GB",
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Mt.", "Jr.", "vs.", "vs", "etc.", "e.g.", "i.e.", "No.", "Inc.",
    "Co.", "Corp.", "Apt.", "Ave.", "Rd.", "min.", "Jan.", "Sept.", "Sep.", "Dec.", "Nov.",
    "$0.01", "$1.01", "£2.01", "€0.01", "$12.10", "4:10", "12:10", "0:10", "123456", "100000", "1050-1080", "1000-1099",
    "999-2001", "2099-2100", "12 Main St.", "123456 Main St.", "1600 Elm Way",
    "January", "March", "May", "December", "Main", "Elm", "Street", "Avenue", "Drive", "Way", "in", "since", "the",
    "zip", "ZIP", "code", "postal", "year", "by", "is", "Patel", "Smith", "Paul", "1990s", "2000s", "'90s",
    # the normalizer audit's forms (validation/audio-assurance-v1/normalizer-audit.md)
    "2024-03-15", "2024-13-01", "+1 415 555 0123", "+44 20 7946 0958", "+1-800", "555.123.4567", "+5", "+$3", "2+2",
    "1=2", "=-3", "5'10\"", "6' 2\"", "5'10”", "5'10''", "1.2.3", "1.2.3.4", "192.168.1.1", "256.1.1.1", "3e8", "6.02e23",
    "2E-5", "1e+4", "10^6", "2^-3", "2^2", "-$3", "$10-$20", "$10 - 20", "$40k", "$5M", "$3.2B", "$2bn", "$3.999",
    "15/03/1990", "31/12/2024", "1:23:45", "0:00:00", "3:30pm", "9:00am", "6:15P.M", "3:1", "16:9", "1:100", "3:05",
    "2-3%", "2.5–3.5 %", "80s", "'00s", "2 1/2", "1 3/4", "12 1/8", "3 5/4", "50/50", "24/7", "9/1", "1/1", "4x4",
    "1920x1080", "2.5×3", "4x4.5a", "10 - 3 = 7", "8 / 2", "5 − 2 =", "1,234.5", "12,345.67", "1.5x", "22B", "3D", "5k",
    "6am", "x", "e", "E", "M", "k", "B", "bn", "^", "'", "\"", "Oak", "Lake", "Then", "Visit", "go", "on", "May", "Dec.",
    "thirty-first,",
]


def fuzz(seed, count):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        parts = [rng.choice(PIECES) for _ in range(rng.randint(1, 14))]
        seps = [rng.choice(["", " ", " ", " ", "  ", ", ", "\n"]) for _ in parts]
        out.append("".join(p + s for p, s in zip(parts, seps)))
    return out


def test_fuzz(tool, tmp_path):
    compare(tool, tmp_path, fuzz(11, 30000))


def test_rules_read_as_intended(tool, tmp_path):
    cases = {
        "It opens at 9:00 AM.": "It opens at nine AM.",
        "Call 911 now.": "Call nine one one now.",
        "The zip code is 12345.": "The zip code is one two three four five.",
        "It was 20° today.": "It was twenty degrees today.",
        "Elm Dr. is closed; see Dr. Who.": "Elm Drive is closed; see Doctor Who.",
        "I live at 12 Oak St. It is quiet.": "I live at twelve Oak Street. It is quiet.",
        "Bring fruit, nuts, etc. Then leave.": "Bring fruit, nuts, et cetera. Then leave.",
        "The year 1999 ended.": "The year nineteen ninety-nine ended.",
        "1500 people came in 1500.": "one thousand five hundred people came in fifteen hundred.",
    }
    assert native(tool, tmp_path, list(cases)) == list(cases.values())


def test_long_text_is_linear(tool, tmp_path):
    long = " ".join(["On Jan. 3, 2025 at 10:30 a.m., Dr. Lee paid $1,234.56 for 12 km of 1990s road, e.g. 45%."] * 4000)
    t0 = time.time()
    got = native(tool, tmp_path, [long])[0]
    assert got == reference.normalize(long) and time.time() - t0 < 30


def test_long_numbers_match_the_reference(tool, tmp_path):
    """cardinal() above 12 digits is built in one pass (audio-assurance-v1): the same words as the reference's
    recursion, for every length to 300 digits, as cardinals, ordinals, money, percentages and grouped numbers."""
    rng = random.Random(5)
    texts = []
    for n in range(1, 301):
        d = str(rng.randrange(1, 10)) + "".join(rng.choice("0123456789") for _ in range(n - 1))
        z = "1" + "0" * (n - 1)  # the zero groups ("... trillion" with nothing after)
        grouped = f"{int(d):,}"
        texts += [f"Count {d} now.", f"The {d}th one.", f"Pay ${d} today.", f"About {d}% of it.", f"Total {grouped} items.",
                  f"Count {z} now.", f"The {z}st one."]
    compare(tool, tmp_path, texts)


@pytest.mark.parametrize("seed", ["1,", ",0", "1,000", "1,1", "1,1%", "$1,", "1.1,", "1,1 km ", ",000", "2024-01-",
                                  "a-2024, 2024 ", "1'1\" ", "+1 1 1 ", "1:1:1 ", "1.1.", "$1-", "1-1%", "1 1/", "1^1"])
def test_hostile_text_is_linear(tool, tmp_path, seed):
    """Inputs that were quadratic (a [\\d,]* scan from every digit after a comma; audio-assurance-v1 red-team): doubling
    the text at most about doubles the time."""
    times = []
    for n in (200_000, 400_000):
        text = (seed * (n // len(seed) + 1))[:n]
        t0 = time.time()
        native(tool, tmp_path, [text])
        times.append(time.time() - t0)
    assert times[1] < 5 and times[1] < 3 * max(times[0], 0.2), times


def test_a_huge_number_does_not_overflow_the_stack(tool, tmp_path):
    """A 400,000-digit ordinal recursed once per 12 digits and crashed the server (audio-assurance-v1 red-team)."""
    for text in ("1" * 400_000 + "st", "$" + "9" * 400_000, "1" * 400_000 + "%"):
        t0 = time.time()
        got = native(tool, tmp_path, [text])[0]
        assert time.time() - t0 < 20 and got.count("trillion") == (400_000 - 1) // 12


# The normalizer audit (validation/audio-assurance-v1/normalizer-audit.md): each fix, and the real-text cases that
# shaped it (wikitext ranges before "= =", "2012 / 13" seasons, coordinates in minutes and seconds), on both builds.
AUDIT_FIXES = {
    "The ratio is 1,234.5.": "The ratio is one thousand two hundred thirty-four point five.",
    "Profits were -$3 million.": "Profits were minus three million dollars.",
    "Prices rose 2-3% this year.": "Prices rose two to three percent this year.",
    "Costs run $10-$20.": "Costs run ten dollars to twenty dollars.",
    "In the '90s and the 80s we danced.": "In the nineties and the eighties we danced.",
    "It grossed $3.2B, then $5M, then $40k.": "It grossed three point two billion dollars, then five million dollars, then forty thousand dollars.",
    "Gas is $3.999 a gallon.": "Gas is three point nine nine nine dollars a gallon.",
    "Call at 3:30pm or 9:00am.": "Call at three thirty pm or nine am.",
    "She ran the half in 1:35:10.": "She ran the half in one hour thirty-five minutes ten seconds.",
    "Mix in a 3:1 ratio; the aspect is 16:9.": "Mix in a three to one ratio; the aspect is sixteen to nine.",
    "Dec. 31, 1999 was a Friday.": "December thirty-first, nineteen ninety-nine was a Friday.",
    "In May 2020, we stayed home.": "In May twenty twenty, we stayed home.",
    "Due 2024-03-15.": "Due March fifteenth, twenty twenty-four.",
    "Born 15/03/1990.": "Born March fifteenth, nineteen ninety.",
    "It is 2 1/2 miles, or 1 1/8.": "It is two and a half miles, or one and an eighth.",
    "A 50/50 split, 24/7, odds of 9/1.": "A fifty fifty split, twenty-four seven, odds of nine to one.",
    "Call 555.123.4567.": "Call five five five, one two three, four five six seven.",
    "Dial +1 415 555 0123.": "Dial plus one, four one five, five five five, oh one two three.",
    "It's 5'10\" tall.": "It's five feet ten inches tall.",
    "Gate 22B, a 5k, 4G, 3D, 6am.": "Gate twenty-two B, a five k, four G, three D, six am.",
    "I drive a 4x4.": "I drive a four by four.",
    "Install version 1.2.3, not 2.4.1.": "Install version one point two point three, not two point four point one.",
    "Connect to 192.168.1.1 now.": "Connect to one nine two dot one six eight dot one dot one now.",
    "10 - 3 = 7 and 8 / 2 = 4.": "ten minus three equals seven and eight divided by two equals four.",
    "It's 2+2, then +5%.": "It's two plus two, then plus five percent.",
    "It's 2^10 or 10^2.": "It's two to the tenth or ten squared.",
    "About 6.02e23 atoms; 2E-5 is tiny.": "About six point zero two times ten to the twenty-third atoms; two times ten to the minus fifth is tiny.",
    # guards
    " = = Operation Tracer : 1941 – 1942 = = ": " equals equals Operation Tracer : nineteen forty-one to nineteen forty-two equals equals ",
    "the 2012 / 13 season": "the two thousand twelve / thirteen season",
    "Longitude, 5 deg. 49' 30\" W.": "Longitude, five deg. forty-nine' thirty\" W.",
    "Visit St. Louis with Sallie St. Clair.": "Visit Saint Louis with Sallie Saint Clair.",
    "In 2024 1/3 of voters came.": "In twenty twenty-four one third of voters came.",
}


def test_audit_fixes(tool, tmp_path):
    assert [reference.normalize(t) for t in AUDIT_FIXES] == list(AUDIT_FIXES.values())
    assert native(tool, tmp_path, list(AUDIT_FIXES)) == list(AUDIT_FIXES.values())


# The red-team round (normalizer-audit.md): each finding's inputs, read as before the audit where a rule could not be
# made right (verses, race times, ports, "9/1" as a date, "$5 - $3 = $2", IDs and links), on both builds.
REDTEAM = {
    # 1: decimal parts after exponents, seconds and ratios are read, never left as digits
    "The square root is 2^0.5 and the growth 10^1.5.": "The square root is two to the zero point five and the growth ten to the one point five.",
    "The density is 1.5e3.5 units.": "The density is one point five times ten to the three point five units.",
    "She won in 2:21:45.3 at Boston.": "She won in two hours twenty-one minutes forty-five point three seconds at Boston.",
    "Shot in a 1:1.85 aspect ratio.": "Shot in a one to one point eight five aspect ratio.",
    "Her split was 1:02:03.45.": "Her split was one oh two:three point four five.",
    "Lap time 1:23.456 at Monza.": "Lap time one twenty-three point four five six at Monza.",
    # 2: plurals keep their "s"
    "The airline retired its 747s and F-16s.": "The airline retired its seven hundred forty-sevens and F-sixteens.",
    "He rolled two 6s; 100s more came.": "He rolled two sixes; hundreds more came.",
    "The under-21s and AR-15s.": "The under-twenty-ones and AR-fifteens.",
    "The 49ers won.": "The forty-nineers won.",
    "Those 4x4s are expensive.": "Those four by fours are expensive.",
    # 3: a clock time with seconds
    "The alarm went off at 07:15:00 sharp.": "The alarm went off at seven fifteen sharp.",
    "Kickoff is at 19:30:00 local time.": "Kickoff is at nineteen thirty local time.",
    "The log shows 14:30:00 UTC.": "The log shows fourteen thirty UTC.",
    # 4: ratios only with a ratio word
    "She ran 27:15.": "She ran twenty-seven:fifteen.",
    "Skip to 45:12 in the podcast.": "Skip to forty-five:twelve in the podcast.",
    "Read Genesis 1:1 and Psalm 119:105.": "Read Genesis one:one and Psalm one hundred nineteen:one hundred five.",
    "Use port 22:80.": "Use port twenty-two:eighty.",
    # 5: prose
    "Of the 300 2/3 voted yes.": "Of the three hundred two thirds voted yes.",
    "Starting 9/1, rent is due on 12/1.": "Starting nine firsts, rent is due on twelve firsts.",
    "I paid $5 - $3 = $2 change.": "I paid five dollars - three dollars equals two dollars change.",
    "The price: £3.50 - £2.00 = £1.50.": "The price: three pounds and fifty pence - two pounds equals one pound and fifty pence.",
    "The index was 100 -3% lower.": "The index was one hundred minus three percent lower.",
    "1+1=2 and 3-1=2.": "one plus one equals two and three minus one equals two.",
    "Week 3 4/5 of games.": "Week three four fifths of games.",
    # 6: IDs, hex, links, versions, dotted dates
    "The UUID 123e4567-e89b-12d3.": "The UUID one hundred twenty-three e4567-e89b-twelve d3.",
    "Set color #3e5; Part 2E5-100; Room 2E1.": "Set color #three e5; Part two E5-one hundred; Room two E1.",
    "Write 0x10 to the register.": "Write zero x10 to the register.",
    "Go to example.com/2024-03-15/post.": "Go to example.com/two thousand twenty-four-zero three-fifteen/post.",
    "Email me at sales-2024-03-15@corp.com.": "Email me at sales-two thousand twenty-four-zero three-fifteen@corp.com.",
    "See http://192.168.1.1:8080/admin now.": "See http://one hundred ninety-two point one six eight.1.1:eight thousand eighty/admin now.",
    "Update to version 4.2.0.1, mask 255.255.255.0, ping 8.8.8.8.": "Update to version four point two point zero point one, mask two five five dot two five five dot two five five dot zero, ping eight dot eight dot eight dot eight.",
    "The meeting is on 15.03.2024; 1.234.567 people.": "The meeting is on March fifteenth, twenty twenty-four; one million two hundred thirty-four thousand five hundred sixty-seven people.",
}


def test_redteam(tool, tmp_path):
    assert [reference.normalize(t) for t in REDTEAM] == list(REDTEAM.values())
    assert native(tool, tmp_path, list(REDTEAM)) == list(REDTEAM.values())


# Red-team round 2 (normalizer-audit.md): each finding, and the regressions the real-text gate found
# (tools/normalizer_gate.py), on both builds.
REDTEAM2 = {
    # 1: comma-grouped numbers are whole numbers, or the rule does not fire
    "A resolution of 2,560×1,440 pixels.": "A resolution of two thousand five hundred sixty by one thousand four hundred forty pixels.",
    "A contrast ratio of 1,000:1.": "A contrast ratio of one thousand:one.",
    "The value 1,000^2 is big.": "The value one thousand^two is big.",
    "Height 1,000'6\" tower.": "Height one thousand'six\" tower.",
    # 2: four valid octets are an address, never a grouped number
    "Ping 192.168.100.100 and 255.255.255.255.": "Ping one nine two dot one six eight dot one zero zero dot one zero zero and two five five dot two five five dot two five five dot two five five.",
    "- 128.0.0.0 to 191.255.255.255": "- one two eight dot zero dot zero dot zero to one nine one dot two five five dot two five five dot two five five",
    "IP address 123.345.128 visited.": "IP address one hundred twenty-three point three hundred forty-five point one hundred twenty-eight visited.",
    # 3: a range to a mixed number
    "Let it rise for 1-1 1/2 hours.": "Let it rise for one to one and a half hours.",
    "Simmer 1 - 1 1/2 hours.": "Simmer one to one and a half hours.",
    "-1 1/2 cups flour": "minus one one half cups flour",
    # 4: a decimal and an "s"
    "The query took 2.5s to run; the 1.5ers.": "The query took two point five s to run; the one point five ers.",
    # 5: h:mm:ss
    "As of 10:15:30 the server is down.": "As of ten fifteen:thirty the server is down.",
    "Start time 9:30:00 a.m. sharp; local time 11:30:00 PM.": "Start time nine thirty a.m. sharp; local time eleven thirty PM.",
    "She finished in 02:15:30.": "She finished in two hours fifteen minutes thirty seconds.",
    "The video runs 00:03:25 in total.": "The video runs three minutes twenty-five seconds in total.",
    "Runtime: 01:45:00.": "Runtime: one hour forty-five minutes.",
    "Results at 24:00:00 exactly.": "Results at twenty-four hours exactly.",
    # 6
    "Plays at 7.30pm and 12.35am.": "Plays at seven thirty pm and twelve thirty-five am.",
    "The 911s.": "The nine one ones.",
    "A 4x4x4 cube and 19.5x17.9x6.3cm.": "A four by four by four cube and nineteen point five by seventeen point nine by six point three cm.",
    "Rated 10/10, 5/5 and 4/4 time.": "Rated ten out of ten, five out of five and four four time.",
    "On 3.15.2024 it shipped.": "On March fifteenth, twenty twenty-four it shipped.",
    "Version 1.100.200 is out.": "Version one point one hundred point two hundred is out.",
    "In March 1234 people protested.": "In March one thousand two hundred thirty-four people protested.",
    "The Dow fell 30 - 2% later.": "The Dow fell thirty - two percent later.",
    # the real-text gate
    "So 5 – 4 = 1.": "So five minus four equals one.",
    "Open 8-9:30am.": "Open eight to nine thirty am.",
    "Doi: 10.1080/15332691.2017.1351050": "Doi: ten point one zero eight zero/fifteen million three hundred thirty-two thousand six hundred ninety-one point two zero one seven.1351050",
    "[-10^9, 10^9] and 1/2^3.": "[minus ten to the ninth, ten to the ninth] and one half^three.",
    "Dewey: the 100s; 100s of millions.": "Dewey: the one hundreds; hundreds of millions.",
}


def test_redteam2(tool, tmp_path):
    assert [reference.normalize(t) for t in REDTEAM2] == list(REDTEAM2.values())
    assert native(tool, tmp_path, list(REDTEAM2)) == list(REDTEAM2.values())


# Red-team round 3 (normalizer-audit.md): n/n in arithmetic, zero-led version parts, money subtraction, on both builds.
REDTEAM3 = {
    "Gap size = 2/2 = 1.": "Gap size equals two halves equals one.",
    "60/60 = 1 and (3/3)*4 and 6/6 + 1.": "sixty sixtieths equals one and (three thirds)*four and six sixths plus one.",
    "He went 3/3; she was 2/2.": "He went three thirds; she was two halves.",
    "Rated 10/10, 5/5 stars; 20/20 vision; 4/4 time; 50/50; 24/7.": "Rated ten out of ten, five out of five stars; twenty twenty vision; four four time; fifty fifty; twenty-four seven.",
    "I saw it on 20/20 last night.": "I saw it on twenty twentieths last night.",
    "Ubuntu 22.04.3 and firmware 1.02.3.": "Ubuntu twenty-two point zero four point three and firmware one point zero two point three.",
    "She had $450 - $75 - $120 = $255 left.": "She had four hundred fifty dollars - seventy-five dollars - one hundred twenty dollars equals two hundred fifty-five dollars left.",
    "Total: $100 - $18 leaves $82.": "Total: one hundred dollars - eighteen dollars leaves eighty-two dollars.",
    "$60-$13-$6-$9": "sixty dollars-thirteen dollars-six dollars-nine dollars",
    "Cost: $5-$7-$9 for S/M/L.": "Cost: five dollars-seven dollars-nine dollars for S/M/L.",
    "$5 - $10 = -$5": "five dollars - ten dollars equals minus five dollars",
    "It costs $5-$10 and $1,000-$2,000.": "It costs five dollars to ten dollars and one thousand dollars to two thousand dollars.",
    "24×7*3 and 3x10(8).": "twenty-four×seven*three and three x10(eight).",
    "The PX-100s.": "The PX-one hundreds.",
    # the real-text gate's round-3 findings
    "b = 20 - 1 = 40 - 1 = <<39": "b equals twenty minus one equals forty to one equals <<thirty-nine",
    "profit=$<<2000000-400000=1600000>>": "profit=$<<two million minus four hundred thousand equals one million six hundred thousand>>",
    "2^31-1 and 2^63 - 1": "two to the thirty-first minus one and two to the sixty-third minus one",
    "Thus, 3.5x=1050.": "Thus, three point five x=one thousand fifty.",
    "A slow 4/4 rhythm in 4/4 meter.": "A slow four four rhythm in four four meter.",
    "2) 24/7 Availability": "two) twenty-four seven Availability",
    "- $10,000 - $20,000": "- ten thousand dollars to twenty thousand dollars",
    "(3/3)*4 and Wednesday (7/7):": "(three thirds)*four and Wednesday (seven sevenths):",
}


def test_redteam3(tool, tmp_path):
    assert [reference.normalize(t) for t in REDTEAM3] == list(REDTEAM3.values())
    assert native(tool, tmp_path, list(REDTEAM3)) == list(REDTEAM3.values())


def rule_inputs(seed=7, count=3000):
    """Inputs the audit's rules read (each form, with decimals, signs, plurals and context words around it)."""
    rng = random.Random(seed)
    n = lambda: str(rng.choice([rng.randint(0, 12), rng.randint(0, 99), rng.randint(100, 2099)]))
    dec = lambda: n() + rng.choice(["", "", "." + str(rng.randint(0, 999))])
    forms = [
        lambda: f"{dec()}^{rng.choice(['', '-'])}{dec()}", lambda: f"{dec()}e{rng.choice(['', '-', '+'])}{dec()}",
        lambda: f"{dec()}E-{dec()}", lambda: f"{rng.randint(0, 30)}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
        + rng.choice(["", "." + str(rng.randint(0, 99))]),
        lambda: f"{n()}:{rng.randint(1, 120)}" + rng.choice(["", "." + str(rng.randint(0, 99))]),
        lambda: f"{n()}s", lambda: f"{rng.randint(1, 12)} {rng.randint(1, 15)}/{rng.choice([2, 3, 4, 8, 16])}",
        lambda: f"{n()}/{n()}", lambda: f"{rng.randint(1, 99)}x{dec()}" + rng.choice(["", "s"]), lambda: f"{n()}.{n()}.{n()}",
        lambda: f"{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}",
        lambda: f"{rng.randint(1, 9)}'{rng.randint(0, 11)}\"", lambda: f"${dec()}{rng.choice(['k', 'M', 'B', 'bn', ''])}",
        lambda: f"${n()}-${n()}", lambda: f"{n()}-{n()}%", lambda: f"+{dec()}%", lambda: f"{n()}+{n()}={n()}",
        lambda: f"{n()} - {n()} = {n()}", lambda: f"{n()}-{n()}={n()}", lambda: f"'{rng.randint(1, 9)}0s",
        lambda: f"{rng.randint(1000, 2099)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        lambda: f"{rng.randint(13, 31)}/{rng.randint(1, 12)}/{rng.randint(1000, 2099)}",
        lambda: f"+1 {rng.randint(200, 999)} {rng.randint(200, 999)} {rng.randint(1000, 9999)}",
    ]
    lead = ["", "at ", "in ", "ran ", "the ratio ", "odds of ", "aspect ", "Ping ", "version ", "It is ", "("]
    tail = ["", ".", ",", " now", " ratio", "!", ")"]
    return [rng.choice(lead) + rng.choice(forms)() + rng.choice(tail) for _ in range(count)]


def test_rule_outputs_hold_no_digit(tool, tmp_path):
    """Whatever a rule reads, no digit is left behind: a decimal part, an exponent, a second, a plural is read in words
    (the red-team found "two to the zeroth.5", "seconds.3")."""
    texts = rule_inputs()
    got = native(tool, tmp_path, texts)
    left = [(t, g) for t, g in zip(texts, got) if re.search(r"\d", g)]
    assert not left, left[:10]
    assert got == [reference.normalize(t) for t in texts]


def test_reference_feet_and_long_numbers():
    """The reference (Python) only: the feet-and-inches angle test was quadratic (64 KiB of 1'1" took 5.4 s); numbers
    over 4300 digits raised ValueError and grouped thousands over a few hundred groups overflowed the recursion."""
    t0 = time.time()
    reference.normalize("1'1\" " * 13_000)
    assert time.time() - t0 < 2
    for text in ("3e" + "1" * 5000, "1/" + "7" * 5000, "555." * 16_000, "2^" + "9" * 5000):
        reference.normalize(text)


# Entries of tests/data/normalizer_audit.json that still differ from a careful reading, as recorded in the audit
# (style, not misleading, or a form left for later: see normalizer-audit.md). Any other non-fine entry fails.
AUDIT_RESIDUAL = {
    "The video is 1:23:45 long.", "The score at halftime was 2:1.",  # the red-team round: as before the audit
    "In May 2020 we stayed home.",  # red-team round 2: a bare month before a lowercase word reads the number as before
    "Go down Oak St. Then turn.", "A 401(k) plan helps.", "The F-150 sells well.", "She flew a Boeing 747.",
    "Use HTTP/2.", "Join #2024goals.", "It's 1 000 000 dollars.", "It is 3e8 m/s.",
    "She ran the 100m in 10.49 seconds.", "A 6' 2\" man.", "My extension is 4021.",
    "It measured .5 inches.", "We have 15,00 items.", "World War II ended.", "Henry VIII had six wives.",
    "Queen Elizabeth II reigned long.", "Super Bowl LVIII was close.", "Read Chapter IV first.",
    "Louis XIV built Versailles.", "Rocky III came out.",
}


def test_audit_corpus(tool, tmp_path):
    spec = importlib.util.spec_from_file_location("normalizer_audit", ROOT / "tools/normalizer_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    entries = json.loads((ROOT / "tests/data/normalizer_audit.json").read_text(encoding="utf-8"))["entries"]
    assert len(entries) >= 300
    not_fine = {e["text"] for e in entries if audit.classify(e, reference.normalize(e["text"])) != "fine"}
    assert not_fine == AUDIT_RESIDUAL
    compare(tool, tmp_path, [e["text"] for e in entries])
