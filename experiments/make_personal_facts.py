"""Generate a pool of natural, first-person personal facts for the QA pipeline eval.

Facts are about ONE user: their own attributes plus attributes of people
around them (sister, boss, best friend...). Each pool entry is a distinct
slot (subject x attribute) so a single ledger can hold all of them without
supersede collisions; `alternatives` gives other plausible values for the
same slot so the eval can stage corrections ("actually, green").

    python3 experiments/make_personal_facts.py            # writes experiments/data/personal_facts.json
    python3 experiments/make_personal_facts.py --seed 7 --out /tmp/pool.json
"""

import argparse
import json
import random
from pathlib import Path

NAMES = ["Mara", "Tobias", "Ines", "Kwame", "Soraya", "Declan", "Priya", "Lukas", "Yuki",
         "Fernanda", "Oluwaseun", "Hanna", "Rafael", "Zainab", "Mateo", "Leila", "Jonas",
         "Amara", "Viktor", "Noor", "Elias", "Sanne", "Tariq", "Greta", "Idris", "Maya",
         "Bruno", "Aiko", "Kasper", "Ngozi", "Teodor", "Camille", "Reza", "Linnea", "Omar"]
CITIES = ["Lisbon", "Porto", "Oslo", "Tallinn", "Kyoto", "Valparaiso", "Ljubljana", "Montreal",
          "Cape Town", "Wellington", "Bologna", "Antwerp", "Krakow", "Cusco", "Hanoi", "Tbilisi",
          "Reykjavik", "Marseille", "Adelaide", "Nairobi", "Seville", "Bergen", "Zagreb",
          "Galway", "Austin", "Vancouver", "Lahore", "Medellin", "Busan", "Riga"]
JOBS = ["pilot", "beekeeper", "radiologist", "cartographer", "pastry chef", "marine biologist",
        "sound engineer", "librarian", "stonemason", "data analyst", "midwife", "locksmith",
        "landscape architect", "translator", "firefighter", "glassblower", "dentist",
        "game designer", "piano tuner", "civil engineer", "florist", "paramedic", "notary",
        "wind turbine technician", "copy editor", "veterinarian", "tailor", "geologist"]
EMPLOYERS = ["Northwind Logistics", "Brightwater Clinic", "Halvorsen Shipyards", "Quillon Press",
             "Meridian Observatory", "Tessera Games", "Alder & Finch", "Copperleaf Energy",
             "Saltmarsh Studios", "Verdant Labs", "Blue Heron Bakery", "Ironwood Foundry",
             "Lumen Analytics", "Harbor Light Library", "Kestrel Aviation", "Orchard Street Dental"]
COLORS = ["teal", "burgundy", "mustard", "lavender", "forest green", "coral", "navy", "ochre",
          "turquoise", "plum", "charcoal", "crimson", "mint", "amber", "indigo", "olive"]
FOODS = ["ramen", "paella", "pierogi", "jollof rice", "bibimbap", "falafel", "moussaka",
         "pho", "tamales", "risotto", "shakshuka", "biryani", "goulash", "ceviche", "dumplings",
         "lasagna", "injera", "pad thai", "bouillabaisse", "feijoada"]
HOBBIES = ["bouldering", "woodturning", "birdwatching", "fencing", "pottery", "sailing",
           "beekeeping", "orienteering", "calligraphy", "kitesurfing", "chess", "archery",
           "baking sourdough", "astrophotography", "salsa dancing", "fly fishing", "knitting",
           "ultimate frisbee", "bonsai", "tango"]
PETS = ["Biscuit", "Pixel", "Mochi", "Rufus", "Olive", "Nimbus", "Waffles", "Juniper",
        "Captain", "Pebble", "Loki", "Clementine", "Bruno", "Saffron", "Tofu", "Marble"]
ALLERGIES = ["peanuts", "shellfish", "penicillin", "latex", "sesame", "bee stings", "soy",
             "tree nuts", "gluten", "eggs", "dust mites", "kiwi"]
LANGUAGES = ["Portuguese", "Japanese", "Swahili", "Norwegian", "Farsi", "Dutch", "Korean",
             "Polish", "Turkish", "Catalan", "Hindi", "Finnish", "Greek", "Vietnamese"]
CARS = ["a yellow Fiat Panda", "a 2012 Volvo wagon", "a red Mazda MX-5", "an electric Kia Niro",
        "a green Land Rover Defender", "a white Toyota Hilux", "a blue Citroen 2CV",
        "a silver Subaru Outback", "a black Mini Cooper", "an orange Honda Element"]
AGES = [str(a) for a in range(23, 67)]
INSTRUMENTS = ["cello", "accordion", "trumpet", "ukulele", "harp", "drums", "oboe",
               "banjo", "saxophone", "mandolin", "bassoon", "theremin"]
TEAMS = ["Benfica", "Ajax", "Celtic", "Boca Juniors", "Fenerbahce", "Kaizer Chiefs",
         "Wellington Phoenix", "Toronto FC", "Urawa Reds", "Rosenborg", "Olympiacos", "Flamengo"]
UNIVERSITIES = ["Uppsala University", "University of Coimbra", "McGill University",
                "University of Cape Town", "Kyoto University", "Trinity College Dublin",
                "University of Bologna", "Charles University", "University of Otago",
                "Universidad de Chile", "Aalto University", "University of Edinburgh"]
DRINKS = ["a flat white", "an oat-milk cappuccino", "a cortado", "black filter coffee",
          "a double espresso", "a matcha latte", "an iced americano", "a chai latte",
          "a long black", "a mocha"]
BOOKS = ["Dune", "The Left Hand of Darkness", "Beloved", "Pachinko", "The Name of the Rose",
         "Kafka on the Shore", "Middlemarch", "Things Fall Apart", "Piranesi", "Solaris",
         "The Shadow of the Wind", "Cloud Atlas"]
SEASONS = ["autumn", "spring", "winter", "summer"]
ANIMALS = ["otters", "red pandas", "octopuses", "owls", "elephants", "axolotls", "wolves",
           "manta rays", "hedgehogs", "penguins"]

# attribute -> (question about {poss}, answer about {subj}, value pool, usage question (user only))
ATTRIBUTES = {
    "name": ("What is {poss} name?", "{subj_cap} name is {v}.", NAMES,
             "Sign off a one-line thank-you note from me using my first name."),
    "city": ("Where does {subj_do} live?", "{subj_cap_s} lives in {v}.", CITIES,
             "Recommend one weekend activity in the city where I live, and name the city."),
    "job": ("What does {subj_do} do for work?", "{subj_cap_s} works as a {v}.", JOBS,
            "Write a one-sentence professional bio for me that states my profession."),
    "employer": ("Where does {subj_do} work?", "{subj_cap_s} works at {v}.", EMPLOYERS,
                 "Draft the first line of an out-of-office reply for me that names my employer."),
    "favorite_color": ("What is {poss} favorite color?", "{subj_cap} favorite color is {v}.", COLORS,
                       "Suggest a paint shade for my bedroom that matches my favorite color, naming the color."),
    "favorite_food": ("What is {poss} favorite food?", "{subj_cap} favorite food is {v}.", FOODS,
                      "Plan tonight's dinner around my favorite food and name the dish."),
    "hobby": ("What is {poss} main hobby?", "{subj_cap} main hobby is {v}.", HOBBIES,
              "Suggest a birthday gift that fits my main hobby and say what the hobby is."),
    "dog": ("What is {poss} dog's name?", "{subj_cap} dog is named {v}.", PETS,
            "Write a one-line birthday wish addressed to my dog by name."),
    "allergy": ("What is {subj_do} allergic to?", "{subj_cap_s} is allergic to {v}.", ALLERGIES,
                "I'm ordering at a restaurant. Which ingredient must I tell them to leave out of my meal?"),
    "language": ("What language does {subj_do} speak besides English?",
                 "{subj_cap_s} speaks {v} besides English.", LANGUAGES,
                 "Say good morning to me in the other language I speak, and name that language."),
    "car": ("What car does {subj_do} drive?", "{subj_cap_s} drives {v}.", CARS,
            "I need to describe my car to a parking attendant. What should I say?"),
    "age": ("How old is {subj_do}?", "{subj_cap_s} is {v} years old.", AGES,
            "How many years until I turn 70? State my current age in your answer."),
    "instrument": ("What instrument does {subj_do} play?", "{subj_cap_s} plays the {v}.", INSTRUMENTS,
                   "Suggest a piece I could practice this week on the instrument I play, naming the instrument."),
    "team": ("What is {poss} favorite sports team?", "{subj_cap} favorite sports team is {v}.", TEAMS,
             "Write a two-word cheer for the team I support, naming the team."),
    "university": ("Where did {subj_do} study?", "{subj_cap_s} studied at {v}.", UNIVERSITIES,
                   "Draft the education line of my resume."),
    "coffee": ("What is {poss} usual coffee order?", "{subj_cap} usual coffee order is {v}.", DRINKS,
               "Place my usual coffee order at the counter for me."),
    "book": ("What is {poss} favorite book?", "{subj_cap} favorite book is {v}.", BOOKS,
             "Recommend a novel similar to my favorite book, and name my favorite."),
    "season": ("What is {poss} favorite season?", "{subj_cap} favorite season is {v}.", SEASONS,
               "Plan a trip for my favorite season and name the season."),
    "animal": ("What is {poss} favorite animal?", "{subj_cap} favorite animal is {v}.", ANIMALS,
               "Pick a zoo exhibit I would enjoy most, naming my favorite animal."),
}

# People around the user. (relation, pronoun-neutral phrase)
RELATIONS = ["sister", "brother", "mother", "father", "best friend", "boss", "neighbor",
             "cousin", "roommate", "colleague", "partner", "son", "daughter"]

# Which attributes make sense for relations (no 'son is 50' etc. — ages are filtered below)
RELATION_ATTRIBUTES = ["name", "city", "job", "favorite_color", "favorite_food", "hobby",
                       "dog", "allergy", "language", "car", "age", "instrument", "team", "coffee", "book"]
RELATION_USAGE = {
    "city": "Which city should I fly to in order to visit my {rel}? Name it.",
    "name": "Write a one-line birthday message addressed to my {rel} by name.",
    "favorite_food": "I'm cooking for my {rel} tonight. Which dish should I make?",
    "hobby": "Suggest a gift for my {rel} that fits their main hobby, naming the hobby.",
    "allergy": "My {rel} is coming to dinner. Which ingredient must I avoid?",
}


def _subject_forms(rel):
    if rel is None:
        return {"poss": "the user's", "subj_cap": "The user's", "subj_cap_s": "The user", "subj_do": "the user"}
    return {"poss": f"the user's {rel}'s", "subj_cap": f"The user's {rel}'s",
            "subj_cap_s": f"The user's {rel}", "subj_do": f"the user's {rel}"}


def build_pool(seed=0, max_alternatives=3):
    rng = random.Random(seed)
    pool = []
    used_names = set()

    def pick(values, avoid=()):
        choices = [v for v in values if v not in avoid]
        return rng.choice(choices)

    for rel in [None] + RELATIONS:
        attrs = ATTRIBUTES if rel is None else {k: ATTRIBUTES[k] for k in RELATION_ATTRIBUTES}
        for attr, (q_t, a_t, values, usage) in attrs.items():
            if rel in ("son", "daughter") and attr in ("job", "employer", "car", "coffee", "university"):
                continue
            forms = _subject_forms(rel)
            if attr == "name":
                v = pick(values, avoid=used_names)
                used_names.add(v)
            elif attr == "age" and rel in ("son", "daughter"):
                v = str(rng.randint(4, 17))
            else:
                v = pick(values)
            alts = [x for x in values if x != v]
            rng.shuffle(alts)
            if attr == "age" and rel in ("son", "daughter"):
                alts = [str(a) for a in rng.sample(range(4, 18), 5) if str(a) != v]
            question = q_t.format(**forms)
            answer = a_t.format(**forms, v=v)
            if rel is None:
                use_q = usage
            else:
                use_q = RELATION_USAGE.get(attr, "").format(rel=rel) or None
            pool.append({
                "slot_id": f"{rel or 'user'}.{attr}".replace(" ", "_"),
                "subject": rel or "user",
                "attribute": attr,
                "question": question,
                "answer": answer,
                "value": v,
                "alternatives": alts[:max_alternatives],
                "usage": use_q,
            })
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "personal_facts.json"))
    args = ap.parse_args()
    pool = build_pool(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n")
    subjects = len({p["subject"] for p in pool})
    with_usage = sum(1 for p in pool if p["usage"])
    print(f"Wrote {len(pool)} facts across {subjects} subjects "
          f"({with_usage} with usage questions) to {args.out}")


if __name__ == "__main__":
    main()
