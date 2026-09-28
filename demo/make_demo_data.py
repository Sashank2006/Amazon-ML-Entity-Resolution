"""
make_demo_data.py - creates a small, FAKE dataset in the same format as the challenge data,
so anyone can run the full pipeline in ~2 minutes without the (non-public) competition files.

The noise imitates what the real data looked like: typos, character swaps (o->0, l->1, s->5),
accents, legal-suffix changes, reordered words, domain-style names, junk words, dropped house
numbers, abbreviations (Road->RD), reordered address parts, empty addresses, replaced (DBA) names,
plus deliberate "hard negatives" (a different business at the same address, or the same name at a
different address). Train contains US + India; test additionally contains France.

Usage:  python demo/make_demo_data.py --out demo_data --n-train 6000 --n-test 3000
"""
import argparse
import os
import random
import string

import pandas as pd

WORDS = {
    "US": ["Summit", "Harbor", "Pioneer", "Maple", "Granite", "Liberty", "Cedar", "Beacon", "Atlas", "Evergreen",
           "Keystone", "Prairie", "Redwood", "Horizon", "Frontier", "Sterling", "Coastal", "Crescent", "Northwind",
           "Silverline", "Bluebird", "Ironwood", "Lakeside", "Riverbend", "Oakmont", "Brightpath", "Canyon",
           "Meadow", "Falcon", "Trinity", "Vista", "Anchor", "Heritage", "Cornerstone", "Legacy", "Juniper",
           "Consulting", "Dental", "Logistics", "Realty", "Roofing", "Bakery", "Fitness", "Studios", "Partners",
           "Holdings", "Ventures", "Solutions", "Systems", "Medical", "Motors", "Capital", "Design", "Foods"],
    "India": ["Sri", "Balaji", "Ganesh", "Lakshmi", "Shakti", "Vinayaka", "Surya", "Om", "Annapurna", "Kaveri",
              "Ganga", "Sai", "Krishna", "Mahalakshmi", "Venkatesh", "Shree", "Durga", "Arihant", "Siddhi",
              "Navya", "Kalpana", "Ashoka", "Bharat", "Deccan", "Malabar", "Coromandel", "Vijaya", "Sunrise",
              "Traders", "Textiles", "Exports", "Infotech", "Constructions", "Enterprises", "Pharma", "Agro",
              "Motors", "Foods", "Chemicals", "Logistics", "Builders", "Industries", "Technologies", "Impex"],
    "France": ["Atelier", "Boulangerie", "Maison", "Jardin", "Lumiere", "Soleil", "Horizon", "Riviere", "Etoile",
               "Voyage", "Cafe", "Pharmacie", "Culture", "Sport", "Sante", "Conseil", "Transports", "Batiment",
               "Services", "Formation", "Ocean", "Montagne", "Vignoble", "Marche", "Studio", "Creation"],
}
STREETS = {
    "US": ["Olive", "Main", "Oak", "Maple", "Willow", "Cedar Ridge", "Hillside", "Lakeview", "Sunset", "River",
           "Pine", "Elm", "Walnut", "Chestnut", "Birch", "Meadow", "Highland", "Park", "Spring", "Valley",
           "Mill", "Church", "Forest", "Harbor", "Orchard", "Prospect", "Franklin", "Jefferson", "Madison"],
    "India": ["Anna Salai", "MG", "Station", "Nehru", "Gandhi", "Tilak", "Patel", "Rajaji", "Temple", "Market",
              "Bazaar", "Lake View", "Cross", "Ring", "Mount", "Residency", "Kamaraj", "Ambedkar", "Sardar"],
    "France": ["de la Paix", "de la Gare", "du Marche", "Victor Hugo", "Pasteur", "Jean Jaures", "des Lilas",
               "du Port", "de l'Eglise", "Nationale", "des Ecoles", "du Moulin", "de la Republique"],
}
STYPES = {"US": [("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Drive", "Dr"), ("Lane", "Ln"),
                 ("Circle", "Cir"), ("Boulevard", "Blvd"), ("Court", "Ct")],
          "India": [("Road", "Rd"), ("Street", "St"), ("Nagar", "Ngr"), ("Colony", "Colony")],
          "France": [("Rue", "R."), ("Avenue", "AV"), ("Boulevard", "BD."), ("Allee", "ALL"), ("Place", "PL")]}
CITIES = {"US": [("Austin", "TX", "Texas"), ("Toledo", "OH", "Ohio"), ("Salem", "OR", "Oregon"),
                 ("Madison", "WI", "Wisconsin"), ("Denver", "CO", "Colorado"), ("Raleigh", "NC", "North Carolina"),
                 ("Tucson", "AZ", "Arizona"), ("Albany", "NY", "New York"), ("Fresno", "CA", "California"),
                 ("Dayton", "OH", "Ohio"), ("Lansing", "MI", "Michigan"), ("Mesa", "AZ", "Arizona")],
          "India": [("Pune", "MH", "Maharashtra"), ("Hyderabad", "TG", "Telangana"), ("Jaipur", "RJ", "Rajasthan"),
                    ("Surat", "GJ", "Gujarat"), ("Chennai", "TN", "Tamil Nadu"), ("Kolkata", "WB", "West Bengal"),
                    ("Lucknow", "UP", "Uttar Pradesh"), ("Bangalore", "KA", "Karnataka"), ("Kochi", "KL", "Kerala")],
          "France": [("Bordeaux", "Gironde", "Nouvelle-Aquitaine"), ("Lille", "Nord", "Hauts-de-France"),
                     ("Nantes", "Loire-Atlantique", "Pays de la Loire"), ("Lyon", "Rhone", "Auvergne-Rhone-Alpes")]}
SUFFIX = {"US": [("LLC", "L.L.C."), ("Inc", "Incorporated"), ("Corp", "Corporation"), ("PLLC", "PLLC"), ("", "")],
          "India": [("Pvt Ltd", "Private Limited"), ("Limited", "Ltd"), ("LLP", "LLP"), ("", "")],
          "France": [("SARL", "S.A.R.L."), ("SAS", "SAS"), ("EURL", "EURL"), ("SCI", "S.C.I."), ("", "")]}
JUNK_FRONT = ["The", "Mr", "Sri", "***", "M/s"]
JUNK_BACK = ["Services", "Center", "Co", "Group", "#41961"]
HOMOGLYPH = {"o": "0", "l": "1", "s": "5", "i": "l", "a": "á", "e": "é"}
FAKE_DBA = ["Nexora", "Vantix", "Zelmora", "Quorvia", "Brivon", "Talvex", "Orinda", "Kovana"]


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def entity(self, c):
        r = self.r
        return dict(c=c, name=" ".join(r.sample(WORDS[c], r.choice([2, 2, 3]))), sfx=r.choice(SUFFIX[c]),
                    num=str(r.randint(1, 40000 if c == "US" else 999)), street=r.choice(STREETS[c]),
                    st=r.choice(STYPES[c]), city=r.choice(CITIES[c]))

    def typo(self, s):
        r = self.r
        if len(s) < 4:
            return s
        i = r.randrange(1, len(s) - 1)
        op = r.random()
        if op < .3:
            return s[:i] + s[i + 1:]
        if op < .6:
            return s[:i] + r.choice(string.ascii_lowercase) + s[i:]
        if op < .8:
            return s[:i] + s[i + 1] + s[i] + s[i + 2:]
        return s[:i] + HOMOGLYPH.get(s[i].lower(), s[i]) + s[i + 1:]

    def name(self, e, clean=False):
        r = self.r
        if clean:
            return (e["name"] + " " + e["sfx"][0]).strip()
        x = r.random()
        if x < .04:
            return r.choice(FAKE_DBA)                                     # replaced / trade name
        if x < .08:
            return e["name"].replace(" ", "").lower() + r.choice([".com", ""])   # domain style
        toks = e["name"].split()
        if r.random() < .3:
            j = r.randrange(len(toks)); toks[j] = self.typo(toks[j])
        if r.random() < .15:
            r.shuffle(toks)
        s = " ".join(toks)
        sf = r.choice(e["sfx"]) if r.random() < .7 else ""
        s = (sf + " " + s) if r.random() < .1 else (s + " " + sf)
        if r.random() < .12:
            s = r.choice(JUNK_FRONT) + " " + s
        if r.random() < .12:
            s = s + " " + r.choice(JUNK_BACK)
        if r.random() < .3:
            s = s.upper()
        elif r.random() < .15:
            s = s.lower()
        return " ".join(s.split())

    def address(self, e, clean=False):
        r = self.r
        c, (st, stab), (city, code, full) = e["c"], e["st"], e["city"]
        num = e["num"]
        street = f"{num} {st} {e['street']}" if c == "France" else f"{num} {e['street']} {st}"
        if clean:
            return ", ".join([street, city, code if c == "US" else full])
        if r.random() < .03:
            return ""
        if r.random() < .5:
            street = street.replace(st, stab)
        if r.random() < .15:
            street = street.replace(num + " ", "", 1)
        elif r.random() < .2:
            street = r.choice(["#", "##", "No ", "H.no ", "00"]) + street
        if r.random() < .1:
            street = self.typo(street)
        parts = [street, city, r.choice([code, full])]
        if r.random() < .15:
            parts.pop(1)
        if r.random() < .15:
            r.shuffle(parts)
        if r.random() < .1:
            parts.insert(1, r.choice(["N/A", "PMB 3237", "Unit 102", "<NULL>"]))
        s = ", ".join(parts)
        return s.upper() if r.random() < .4 else s


def build(split, n_s1, countries, seed, out):
    g = Gen(seed)
    r = g.r
    rows = {1: [], 2: [], 3: []}
    gt = []
    ids = iter(r.sample(range(10_000_000, 999_999_999), n_s1 * 14))
    for _ in range(n_s1):
        c = r.choice(countries)
        e = g.entity(c)
        sid = f"S1-{next(ids)}"
        rows[1].append((sid, g.name(e, True), g.address(e, True), c))
        k = 0 if r.random() < .06 else r.choice([1, 2, 3, 3, 4, 4, 5, 6])
        matched = []
        for _ in range(k):
            src = r.choice([2, 3])
            mid = f"S{src}-{next(ids)}"
            rows[src].append((mid, g.name(e), g.address(e), c))
            matched.append(mid)
        gt.append((sid, ",".join(matched)))
        if r.random() < .3:   # hard negative: another business at the same address
            e2 = g.entity(c); e2.update(num=e["num"], street=e["street"], city=e["city"], st=e["st"])
            src = r.choice([2, 3]); rows[src].append((f"S{src}-{next(ids)}", g.name(e2), g.address(e2), c))
        if r.random() < .3:   # hard negative: same name, different address
            e2 = g.entity(c); e2["name"] = e["name"]
            src = r.choice([2, 3]); rows[src].append((f"S{src}-{next(ids)}", g.name(e2), g.address(e2), c))
    for _ in range(n_s1):     # distractors that belong to no Source-1 business
        c = r.choice(countries); e = g.entity(c); src = r.choice([2, 3])
        rows[src].append((f"S{src}-{next(ids)}", g.name(e), g.address(e), c))
    d = os.path.join(out, split)
    os.makedirs(d, exist_ok=True)
    cols = ["entity_id", "business_name", "business_address", "country"]
    for s in (1, 2, 3):
        pd.DataFrame(rows[s], columns=cols).sample(frac=1, random_state=s).to_csv(
            os.path.join(d, f"{split}_source{s}.tsv"), sep="\t", index=False)
    if split == "train":
        pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
            os.path.join(d, "train_ground_truth.tsv"), sep="\t", index=False)
    else:  # keep the answers for the demo's final score (the real challenge hides these)
        pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
            os.path.join(out, "test_answers_DEMO_ONLY.tsv"), sep="\t", index=False)
    print(f"{split}: S1={len(rows[1]):,}  S2={len(rows[2]):,}  S3={len(rows[3]):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_data")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=3000)
    a = ap.parse_args()
    build("train", a.n_train, ["US", "India"], 1, a.out)
    build("test", a.n_test, ["US", "India", "France"], 2, a.out)
