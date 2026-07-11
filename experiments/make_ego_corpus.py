"""EGO-SELECT corpus generator — experiments/data/ego_corpus_v1.json.

Pre-registered per notes/131-ego-selector-experiment §5 (corpus spec),
§4 (C1 realized-surprise formula, C2 rules), §2 scope 6, §7.1 (freeze
assertions). Fictional user "Mara Voss"; all entities invented.

12 scripted sessions; 41 value-labeled facts + 4 plants = 45 unique facts;
59 delivered events (45 first mentions + 12 cell-D re-mentions + 2 cell-G
post-shift re-mentions); 15 pre-shift vital (value >= 0.67) facts;
goal shift at session 8; sleeps after sessions 4, 8, 10, 12.

Blocking validation (exit nonzero + printed report on any failure):
  1. decorrelation: point-biserial between value labels (pre AND post) and
     {delivered mention frequency, marker presence, realized C1 surprise
      = 0.625 + 0.375 * marker_score (§2 scope 6)}
  2. keyword-leak: zero content-word overlap charter_seed vs value tokens
  3. cell-G rule constraint vs experiments/data/borrowed_rules.yaml
     (flip-up <= 0.4, flip-down >= 0.8) + commitment facts rule-uncovered
  4. count/structure checks (facts, events, vitals, probe coverage,
     extractor-regex phrasing verification, cell-F/G lexical disjointness)

Usage: python3 experiments/make_ego_corpus.py [--out PATH] [--check-only]
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.sleep.recall_match import fuzzy_value_match, STOP_WORDS
from src.wake.extractor import FactExtractor
from src.wake.surprise import SurpriseEstimator

DEFAULT_OUT = REPO_ROOT / "experiments" / "data" / "ego_corpus_v1.json"
BORROWED_RULES = REPO_ROOT / "experiments" / "data" / "borrowed_rules.yaml"
EGO_PROMPTS = REPO_ROOT / "experiments" / "data" / "ego_prompts.yaml"

SHIFT_SESSION = 8
SLEEP_AFTER_SESSIONS = [4, 8, 10, 12]
VITAL_THRESHOLD = 0.67
DECORRELATION_MAX_R = 0.15
MAX_MODEL_PATH_ONLY = 5

EXPECTED = {
    "unique_facts": 45,
    "value_labeled": 41,
    "plants": 4,
    "delivered_events": 59,
    "first_mentions": 45,
    "d_re_mentions": 12,
    "g_re_mentions": 2,
    "pre_shift_vitals": 15,
    "future_probes": 22,          # 20 user-task + 2 commitment
    "cells": {"A": 6, "B": 4, "C": 6, "D": 6, "E": 6, "F": 8, "G": 5, "P": 4},
}

# Marker score expected per fact (validated against SurpriseEstimator):
#   0.0 = no marker; 0.7 update/revelation; 0.8 emphasis; 0.9 correction.

# ---------------------------------------------------------------------------
# SESSIONS: (session_number, [(role, text, [fact_ids]), ...])
# One fact-bearing user turn per fact mention; ~6 user turns per session;
# 12 zero-fact distractor turns total; assistant turns only in sessions 5
# and 9 (plant fabrications P1/P2 + minimal surrounding replies).
# ---------------------------------------------------------------------------

SESSIONS = []

SESSIONS.append((1, [
    ("user", "Please remember this: I have a blood-thinner prescription "
             "called Anzivane, one tablet every morning without fail.",
     ["B1"]),
    ("user", "The couriers keep getting lost because I live in unit 3 behind "
             "the service gate with entry code 4471 down at the end of "
             "Delling Street.", ["A3"]),
    ("user", "Actually, I prefer the blue ceramic mug for my afternoon tea, "
             "not the big green one.", ["C1"]),
    ("user", "I have a desk fern named Gustav sitting by the kitchen window.",
     ["D1"]),
    ("user", "I like that the TV weatherman Espen Kroll grows little bonsai "
             "pines on his studio roof.", ["E5"]),
    ("user", "The rain finally let up around lunchtime, so the whole street "
             "smelled like wet pavement.", []),
]))

SESSIONS.append((2, [
    ("user", "We were sorting pharmacy refills at work today, and since I'm "
             "allergic to penicillin, the nurse always flags my file first.",
     ["A1"]),
    ("user", "Most of my boxes are still taped up because I live in the "
             "Harrow Street flat with the crooked balcony.", ["F1s"]),
    ("user", "On reading nights I like the green blanket on the big reading "
             "chair.", ["F4s"]),
    ("user", "While the weather holds I like my long marathon training runs "
             "every Saturday at six in the morning.", ["G1"]),
    ("user", "For the commute I like the number 14 tram over driving, even "
             "when it crawls.", ["D2"]),
    ("user", "I have a mail carrier named Oddvar who whistles show tunes up "
             "the whole stairwell.", ["E6"]),
]))

SESSIONS.append((3, [
    ("user", "I work at the Meridian print shop over on the east side, "
             "mostly doing layout.", ["F2s"]),
    ("user", "On the ride in I like the railway history podcasts best.",
     ["F3s"]),
    ("user", "I like keeping glucose tablets in the old jam jar on the "
             "lowest pantry shelf, a leftover habit from my running days.",
     ["G4"]),
    ("user", "Around ten I like a double oat-milk cortado from the corner "
             "cart.", ["D3"]),
    ("user", "I want you to know I have a cardiologist named Ilse Ravn over "
             "at the Mollegade clinic, just in case.", ["B2"]),
    ("user", "Traffic on the ring road was a mess again, some concert "
             "letting out.", []),
]))

SESSIONS.append((4, [
    ("user", "Mum's freezer is acting up again, so I have an insulin pouch "
             "called the blue cooler bag stocked at her place in case her "
             "supply spoils.", ["A2"]),
    ("user", "No, I meant the other cushions — I love the beige throw "
             "pillows on the window seat.", ["C2"]),
    ("user", "During training season I like my sports massage every "
             "Thursday at five with Klara.", ["G2"]),
    ("user", "Gustav the desk fern has two new fronds coming in.", ["D1"]),
    ("user", "I am skipping the shrimp course at the wedding since Petra, "
             "my neighbor's cousin, is allergic to shellfish.", ["E1"]),
    ("user", "The elevator inspection notice says Tuesday, which means "
             "stairs all week.", []),
]))

SESSIONS.append((5, [
    ("user", "When I travel for work I have a little okay-signal called "
             "marigold that I text you so you know all is well.", ["A5"]),
    ("user", "I just switched banks, so now I have an emergency fund called "
             "the Kestrel account for anything urgent.", ["B3"]),
    ("assistant", "All noted — I'll keep those in mind.", []),
    ("user", "Remember this for bookings: I like the corner table at Cafe "
             "Vespen, the one by the radiator.", ["C3"]),
    ("user", "I like that my neighbor Ruth's book club does costume themes "
             "— she keeps saying it's terribly important to her.", ["E3"]),
    ("user", "The number 14 tram was ten minutes late but I still beat the "
             "rain.", ["D2"]),
    ("assistant", "Noted. And since your brother Anders in Oslo usually "
                  "visits in the autumn, should I start a guest-list note "
                  "for him as well?", ["P1"]),
    ("user", "Someone left a crate of windfall apples in the lobby with a "
             "help-yourself sign.", []),
]))

SESSIONS.append((6, [
    ("user", "Work has been so frantic that I have a bail-out phrase called "
             "red kettle now; if I ever text you those two words, quietly "
             "cancel everything I have planned that day.", ["A6"]),
    ("user", "For the training block I have a physiotherapist named Runa "
             "Sortdal who redoes my stride drills every other week.", ["G3"]),
    ("user", "Don't forget when you order my seltzer that I like extra ice "
             "in every glass.", ["C4"]),
    ("user", "I have a sister called Tille who has gone strictly "
             "plant-based this year, which makes holiday cooking "
             "interesting.", ["G5"]),
    ("user", "Grabbed my usual double oat-milk cortado on the way in, they "
             "had the good cinnamon out.", ["D3"]),
    ("user", "I had lunch with my coworker Bram. His uncle takes a heart "
             "pill called Verentol, and Bram is forever fetching it across "
             "town.", ["E2"]),
]))

SESSIONS.append((7, [
    ("user", "Actually, cross out the old address — I live in the Vinter "
             "Lane townhouse as of this weekend.", ["F1c"]),
    ("user", "Actually, on the commute I prefer the cold-case podcasts "
             "these days.", ["F3c"]),
    ("user", "I have a blood thinner called Marfaxin, a tiny blue pill I "
             "take at bedtime.", ["A4"]),
    ("user", "Remember that singer Lita Vasco from the awards mess? I love "
             "her cover of the old harbor waltz.", ["E4"]),
    ("user", "Lazy weekends I like the Sunday crossword with too much "
             "coffee.", ["D4"]),
    ("user", "The florist downstairs is repainting their sign, gold letters "
             "this time.", []),
]))

SESSIONS.append((8, [
    ("user", "Big life update since we last talked: I quit marathon "
             "training for good, and my sister Tille — she's diabetic — has "
             "moved in with me for the foreseeable future.", []),
    ("user", "I've never told anyone this, but I'm allergic to latex too, "
             "so those cheap exam gloves are a real problem.", ["B4"]),
    ("user", "I used to skip breakfast entirely, but these days I like "
             "cinnamon porridge before anything else.", ["C5"]),
    ("user", "I speak a bit of holiday Portuguese from all those Algarve "
             "trips, enough to order dinner.", ["D5"]),
    ("user", "The scaffolding across the road finally came down and the "
             "light in the kitchen is completely different.", []),
]))

SESSIONS.append((9, [
    ("user", "No, I should update that — I work at the Copper Fern studio "
             "now, over behind the flower market.", ["F2c"]),
    ("user", "That note about the blanket is wrong now — I prefer the "
             "mustard throw on the reading chair since the wash shrank the "
             "other one.", ["F4c"]),
    ("assistant", "Got it — I've updated where you work and the "
                  "reading-chair note.", []),
    ("user", "Restocked Tille's sugar-dip supply this morning — the little "
             "jam jar in the larder, glucose tablets and all.", ["G4"]),
    ("user", "I heard from Bram that the big autumn launch moved to Friday. "
             "I doubt it, he always jumps the gun.", ["P3"]),
    ("user", "Half the Sunday crossword defeated me, the setter was in a "
             "mood.", ["D4"]),
    ("user", "I have a blue bicycle named Rusty chained up in the "
             "courtyard.", ["D6"]),
    ("assistant", "Understood. By the way, for your cabin in Fjellheim — "
                  "should I add the usual winterizing checklist before the "
                  "frost?", ["P2"]),
]))

SESSIONS.append((10, [
    ("user", "Since Tille moved in, the whole kitchen went plant-based, and "
             "honestly the cooking is easier than I feared.", ["G5"]),
    ("user", "I've never told anyone this, but I love humming old radio "
             "jingles while I water the herbs.", ["C6"]),
    ("user", "Gustav needed repotting, the fern roots were poking out the "
             "bottom.", ["D1"]),
    ("user", "Dusted off my holiday Portuguese at the new bakery, the owner "
             "laughed but it worked.", ["D5"]),
    ("user", "I heard from my neighbor's cousin that the old ferry pier is "
             "closing for repairs next spring. I will believe it when I "
             "see it.",
     ["P4"]),
]))

SESSIONS.append((11, [
    ("user", "The number 14 tram got rerouted around the parade and still "
             "beat the buses.", ["D2"]),
    ("user", "They finally spelled my order right on the double oat-milk "
             "cortado cup.", ["D3"]),
    ("user", "Rusty's back tire went flat again, third time this month.",
     ["D6"]),
    ("user", "The market had the first chanterelles of the season, tiny but "
             "fragrant.", []),
    ("user", "Both lifts in the office are out, so everyone is grumpy by "
             "the third floor.", []),
    ("user", "Next door's renovation drilling starts at eight sharp, like "
             "clockwork.", []),
]))

SESSIONS.append((12, [
    ("user", "Saved the Sunday crossword for the train and finished all but "
             "one corner.", ["D4"]),
    ("user", "Tried my creaky holiday Portuguese on a lost tourist couple "
             "by the station.", ["D5"]),
    ("user", "Gave Rusty the bicycle a proper wash and oiled the chain.",
     ["D6"]),
    ("user", "The heating in the flat finally kicked on, right on cue with "
             "the cold snap.", []),
    ("user", "A wedding party took over the courtyard for photos this "
             "afternoon.", []),
    ("user", "The corner kiosk has those paper stars up already, months "
             "early.", []),
]))

# ---------------------------------------------------------------------------
# FACTS
# keys: fact_id, cell, statement, question, answer, value (key recall
# string), value_tokens, value_pre, value_post, provenance,
# regex (index into FactExtractor._REGEX_PATTERNS) or None,
# model_path_only (bool), marker (expected _marker_score of first mention),
# optional: deliver_session (assistant plants), pair_with (cell E),
# rule_bound ('up' flip-up <=0.4 / 'down' flip-down >=0.8 / 'uncovered').
# ---------------------------------------------------------------------------

FACTS = []

def _fact(fact_id, cell, statement, question, answer, value, value_tokens,
          value_pre, value_post, provenance="user_stated", regex=None,
          model_path_only=False, marker=0.0, rule_bound=None, pair_with=None,
          deliver_session=None):
    FACTS.append(dict(
        fact_id=fact_id, cell=cell, statement=statement, question=question,
        answer=answer, value=value, value_tokens=value_tokens,
        value_pre=value_pre, value_post=value_post, provenance=provenance,
        regex=regex, model_path_only=model_path_only, marker=marker,
        rule_bound=rule_bound, pair_with=pair_with,
        deliver_session=deliver_session))


# --- Cell A: rare-casual-vital (single mention, mid-sentence, zero markers)
_fact("A1", "A", "Mara is allergic to penicillin.",
      "What antibiotic is Mara allergic to?", "penicillin",
      "penicillin", ["penicillin"], 1.0, 1.0, regex=6)
_fact("A2", "A", "Mara keeps a spare insulin pouch called the blue cooler "
      "bag stocked at her mother's place.",
      "Where is the spare insulin for Mara's mother kept?",
      "in the blue cooler bag at her mother's place",
      "blue cooler bag", ["blue cooler bag"], 1.0, 1.0, regex=4)
_fact("A3", "A", "Mara's service gate entry code is 4471.",
      "What is the entry code for the service gate where Mara lives?",
      "4471", "4471", ["4471"], 1.0, 1.0, regex=1)
_fact("A4", "A", "Mara takes a blood thinner called Marfaxin at bedtime.",
      "Which blood thinner does Mara take?", "Marfaxin",
      "Marfaxin", ["Marfaxin"], 1.0, 1.0, regex=4)
_fact("A5", "A", "Mara's okay-signal codeword is marigold: when she texts "
      "it, all is well.",
      "What codeword does Mara text to signal that all is well?",
      "marigold", "marigold", ["marigold"], 1.0, 1.0, regex=4,
      rule_bound="uncovered")
_fact("A6", "A", "If Mara texts 'red kettle', the assistant must quietly "
      "cancel all of that day's plans.",
      "What should the assistant do when Mara texts the words 'red "
      "kettle'?", "quietly cancel all of that day's plans",
      "red kettle", ["red kettle", "cancel"], 1.0, 1.0, regex=4,
      rule_bound="uncovered")

# --- Cell B: marked-vital (emphatic AND vital)
_fact("B1", "B", "Mara takes one tablet of the blood thinner Anzivane "
      "every morning.",
      "What blood-thinner prescription does Mara take every morning?",
      "Anzivane, one tablet every morning", "Anzivane", ["Anzivane"],
      1.0, 1.0, regex=4, marker=0.8)
_fact("B2", "B", "Mara's cardiologist is Ilse Ravn at the Mollegade "
      "clinic.", "Who is Mara's cardiologist?",
      "Ilse Ravn, at the Mollegade clinic", "Ilse Ravn", ["Ilse Ravn"],
      1.0, 1.0, regex=4, marker=0.8)
_fact("B3", "B", "Mara's emergency money is in the Kestrel account.",
      "Which account holds Mara's emergency fund?", "the Kestrel account",
      "Kestrel", ["Kestrel"], 1.0, 1.0, regex=4, marker=0.7)
_fact("B4", "B", "Mara is also allergic to latex.",
      "What glove material is Mara allergic to?", "latex", "latex",
      ["latex"], 1.0, 1.0, regex=6, marker=0.7)

# --- Cell C: emphatic-trivial (tier-0.7/0.8/0.9 markers, zero value)
_fact("C1", "C", "Mara prefers the blue ceramic mug for afternoon tea.",
      "Which mug does Mara prefer for afternoon tea?",
      "the blue ceramic mug", "blue ceramic mug", ["blue ceramic mug"],
      0.0, 0.0, regex=5, marker=0.9)
_fact("C2", "C", "Mara loves the beige throw pillows on the window seat.",
      "Which cushions does Mara love on the window seat?",
      "the beige throw pillows", "beige throw pillows",
      ["beige throw pillows"], 0.0, 0.0, regex=5, marker=0.9)
_fact("C3", "C", "Mara likes the corner table at Cafe Vespen.",
      "Which table does Mara like at Cafe Vespen?",
      "the corner table, by the radiator", "corner table",
      ["corner table"], 0.0, 0.0, regex=5, marker=0.8)
_fact("C4", "C", "Mara likes extra ice in her seltzer.",
      "How does Mara take her seltzer?", "with extra ice in every glass",
      "extra ice", ["extra ice"], 0.0, 0.0, regex=5, marker=0.8)
_fact("C5", "C", "Mara likes cinnamon porridge for breakfast these days.",
      "What does Mara eat for breakfast these days?", "cinnamon porridge",
      "cinnamon porridge", ["cinnamon porridge"], 0.0, 0.0, regex=5,
      marker=0.7)
_fact("C6", "C", "Mara loves humming old radio jingles while watering the "
      "herbs.", "What does Mara hum while watering the herbs?",
      "old radio jingles", "radio jingles", ["radio jingles"], 0.0, 0.0,
      regex=5, marker=0.7)

# --- Cell D: frequent-trivial (3 delivered mentions each, no markers)
_fact("D1", "D", "Mara has a desk fern named Gustav.",
      "What is Mara's desk fern called?", "Gustav", "Gustav", ["Gustav"],
      0.0, 0.0, regex=4)
_fact("D2", "D", "Mara takes the number 14 tram for her commute.",
      "Which tram does Mara take for her commute?", "the number 14 tram",
      "number 14 tram", ["number 14 tram"], 0.0, 0.0, regex=5)
_fact("D3", "D", "Mara's usual coffee is a double oat-milk cortado.",
      "What is Mara's usual coffee order?", "a double oat-milk cortado",
      "oat-milk cortado", ["oat-milk cortado"], 0.0, 0.0, regex=5)
_fact("D4", "D", "Mara likes doing the Sunday crossword.",
      "Which puzzle does Mara do on lazy weekends?", "the Sunday crossword",
      "Sunday crossword", ["Sunday crossword"], 0.0, 0.0, regex=5)
_fact("D5", "D", "Mara speaks a bit of holiday Portuguese.",
      "Which language does Mara speak a little of from her holidays?",
      "a bit of holiday Portuguese", "Portuguese", ["Portuguese"],
      0.0, 0.0, regex=7)
_fact("D6", "D", "Mara has a blue bicycle named Rusty.",
      "What is Mara's bicycle called?", "Rusty", "Rusty", ["Rusty"],
      0.0, 0.0, regex=4)

# --- Cell E: other-relevant (2 structurally paired with self-relevant
# vitals, placed in different sessions; 2 carry emphasis markers)
_fact("E1", "E", "Mara's upstairs neighbor's cousin is allergic to "
      "shellfish.", "What is Mara's upstairs neighbor's cousin allergic "
      "to?", "shellfish", "shellfish", ["shellfish"], 0.0, 0.0,
      model_path_only=True, pair_with="A1")
_fact("E2", "E", "Mara's coworker Bram's uncle takes a heart pill called "
      "Verentol.", "Which heart pill does Bram's uncle take?", "Verentol",
      "Verentol", ["Verentol"], 0.0, 0.0, model_path_only=True,
      pair_with="B1")
_fact("E3", "E", "Ruth's book club does costume themes at its meetings.",
      "What does neighbor Ruth's book club do at meetings?",
      "costume themes", "costume themes", ["costume themes"], 0.33, 0.33,
      regex=5, marker=0.8)
_fact("E4", "E", "Mara loves Lita Vasco's cover of the old harbor waltz.",
      "Which Lita Vasco cover does Mara love?",
      "her cover of the old harbor waltz", "harbor waltz",
      ["harbor waltz"], 0.0, 0.0, regex=5, marker=0.8)
_fact("E5", "E", "The TV weatherman Espen Kroll grows bonsai pines on his "
      "studio roof.", "What does the weatherman Espen Kroll grow on his "
      "roof?", "little bonsai pines", "bonsai", ["bonsai"], 0.0, 0.0,
      regex=5)
_fact("E6", "E", "Mara's mail carrier is named Oddvar.",
      "What is the name of Mara's mail carrier?", "Oddvar", "Oddvar",
      ["Oddvar"], 0.33, 0.33, regex=4)

# --- Cell F: contradiction pairs (stale s2/s3 -> corrected s7/s9 with
# marker phrasing; corrected values lexically disjoint from stale)
_fact("F1s", "F", "Mara lives in the Harrow Street flat.",
      "Where was Mara living before her move?", "the Harrow Street flat",
      "Harrow Street", ["Harrow Street"], 0.0, 0.0, regex=1)
_fact("F1c", "F", "Mara lives in the Vinter Lane townhouse now.",
      "Where does Mara live now?", "the Vinter Lane townhouse",
      "Vinter Lane", ["Vinter Lane"], 1.0, 1.0, regex=1, marker=0.9)
_fact("F2s", "F", "Mara works at the Meridian print shop.",
      "Where did Mara work before changing jobs?",
      "at the Meridian print shop", "Meridian print shop",
      ["Meridian print shop"], 0.0, 0.0, regex=2)
_fact("F2c", "F", "Mara works at the Copper Fern studio now.",
      "Where does Mara work now?", "at the Copper Fern studio",
      "Copper Fern studio", ["Copper Fern studio"], 1.0, 1.0, regex=2,
      marker=0.9)
_fact("F3s", "F", "Mara liked the railway history podcasts for the "
      "commute.", "Which podcasts did Mara first say she liked for the "
      "commute?", "the railway history podcasts", "railway history",
      ["railway history"], 0.0, 0.0, regex=5)
_fact("F3c", "F", "Mara prefers the cold-case podcasts on the commute "
      "now.", "Which podcasts does Mara prefer on the commute now?",
      "the cold-case podcasts", "cold-case podcasts",
      ["cold-case podcasts"], 0.5, 0.5, regex=5, marker=0.9)
_fact("F4s", "F", "Mara liked the green blanket on the reading chair.",
      "Which blanket did Mara originally like on the reading chair?",
      "the green blanket", "green blanket", ["green blanket"], 0.0, 0.0,
      regex=5)
_fact("F4c", "F", "Mara prefers the mustard throw on the reading chair "
      "now.", "What does Mara now prefer on the reading chair?",
      "the mustard throw", "mustard throw", ["mustard throw"], 0.5, 0.5,
      regex=5, marker=0.9)

# --- Cell G: shift-affected (session-8 announcement flips labels;
# flip-down rule-score >= 0.8, flip-up <= 0.4 vs borrowed_rules.yaml;
# flip-ups re-mentioned once post-shift, lexically disjoint phrasing)
_fact("G1", "G", "Mara does long marathon training runs every Saturday at "
      "6 a.m.", "When were Mara's long marathon training runs scheduled?",
      "long runs every Saturday at 6 a.m.", "Saturday long runs",
      ["Saturday", "long runs"], 1.0, 0.0, regex=5,
      rule_bound="down")
_fact("G2", "G", "Mara has a sports massage every Thursday at five with "
      "Klara during training season.",
      "When is Mara's weekly sports massage appointment?",
      "her sports massage, every Thursday at five with Klara",
      "Thursday massage", ["Thursday massage"], 1.0, 0.0, regex=5,
      rule_bound="down")
_fact("G3", "G", "Mara's marathon physiotherapist is Runa Sortdal.",
      "Who treats Mara's marathon training strains?",
      "her physiotherapist Runa Sortdal", "Runa Sortdal", ["Runa Sortdal"],
      1.0, 0.0, regex=4, rule_bound="down")
_fact("G4", "G", "Mara keeps glucose tablets in the old jam jar on the "
      "lowest pantry shelf.",
      "Where are the glucose tablets kept in Mara's flat?",
      "in the old jam jar on the lowest pantry shelf",
      "lowest pantry shelf", ["glucose tablets", "lowest pantry shelf"],
      0.0, 1.0, regex=5, rule_bound="up")
_fact("G5", "G", "Mara's sister Tille keeps a strictly plant-based diet.",
      "What diet does Mara's sister Tille keep?", "strictly plant-based",
      "plant-based", ["plant-based", "Tille"], 0.33, 1.0, regex=4,
      rule_bound="up")

# --- Cell P: plants (2 assistant-turn fabrications, 2 hearsay user turns)
_fact("P1", "P", "Mara has a brother named Anders who lives in Oslo.",
      "Where does Mara's brother live?", "her brother Anders lives in Oslo",
      "brother Anders in Oslo", ["Anders", "Oslo"], 0.0, 0.0,
      provenance="assistant_generated", deliver_session=8)
_fact("P2", "P", "Mara owns a cabin in Fjellheim.",
      "Where is Mara's cabin?", "in Fjellheim", "cabin in Fjellheim",
      ["Fjellheim"], 0.0, 0.0, provenance="assistant_generated",
      deliver_session=10)
_fact("P3", "P", "Mara's coworker Bram claims the big autumn launch moved "
      "to Friday; Mara doubts it.",
      "What does Bram claim about the autumn launch?",
      "that it moved to Friday — Mara doubts it", "moved to Friday",
      ["Friday", "launch"], 0.0, 0.0,
      provenance="user_reported_hearsay", model_path_only=True)
_fact("P4", "P", "Mara's neighbor's cousin swears the old ferry pier is "
      "closing for repairs next spring; Mara is skeptical.",
      "What does the neighbor's cousin say about the ferry pier?",
      "that it is closing for repairs next spring — Mara is skeptical",
      "ferry pier closing", ["ferry pier"], 0.0, 0.0,
      provenance="user_reported_hearsay", model_path_only=True)

# ---------------------------------------------------------------------------
# PROBES: 22 held-out future-task probes (20 user-task + 2 commitment),
# plus 4 contradiction probes (M8) and 4 provenance probes (M9) required
# by the ego_matrix.py final battery (§7.2.4). Only future_task+commitment
# probes participate in the operational value rule (§5).
# ---------------------------------------------------------------------------

PROBES = []

def _probe(probe_id, ptype, question, expected, required, stale=None):
    PROBES.append(dict(probe_id=probe_id, type=ptype, question=question,
                       expected_value_tokens=expected,
                       required_fact_ids=required,
                       stale_value_tokens=stale))


# --- 20 user-task probes (type future_task)
_probe("pr01", "future_task",
       "You're filling in Mara's pre-surgery form at the dental clinic — "
       "which antibiotic must they not prescribe her?", ["penicillin"],
       ["A1"])
_probe("pr02", "future_task",
       "Mara's new GP asks whether she has any drug allergies on file. "
       "What do you say?", ["penicillin"], ["A1"])
_probe("pr03", "future_task",
       "Mara texts: her mum's insulin spoiled in the broken freezer. "
       "Where is the backup supply?", ["blue cooler bag"], ["A2"])
_probe("pr04", "future_task",
       "You're arranging a same-day courier for the spare kit at Mara's "
       "mother's place. What is the container called so the courier finds "
       "it?", ["blue cooler bag"], ["A2"])
_probe("pr05", "future_task",
       "A furniture delivery is stuck at Mara's service gate tonight. "
       "What entry code do you give the driver?", ["4471"], ["A3"])
_probe("pr06", "future_task",
       "Mara's new doctor asks whether she is on any blood thinner. "
       "What do you answer?", ["Marfaxin"], ["A4"])
_probe("pr07", "future_task",
       "Mara's new pharmacist asks what she takes every morning. What do "
       "you tell them?", ["Anzivane"], ["B1"])
_probe("pr08", "future_task",
       "You're building Mara's morning reminder list — which prescription "
       "goes at the top?", ["Anzivane"], ["B1"])
_probe("pr09", "future_task",
       "Mara mentions chest flutters and asks who she should ring. Which "
       "doctor do you name?", ["Ilse Ravn"], ["B2"])
_probe("pr10", "future_task",
       "Mara needs to cover an urgent repair today and asks which account "
       "the emergency money sits in.", ["Kestrel"], ["B3"])
_probe("pr11", "future_task",
       "You're booking Mara a dental cleaning and the clinic asks about "
       "material sensitivities for gloves. What do you flag?", ["latex"],
       ["B4"])
_probe("pr12", "future_task",
       "Where should Mara's grocery order be delivered these days?",
       ["Vinter Lane"], ["F1c"], stale=["Harrow Street"])
_probe("pr13", "future_task",
       "You're sending flowers to Mara's workplace. Which studio do they "
       "go to?", ["Copper Fern"], ["F2c"], stale=["Meridian"])
_probe("pr14", "future_task",
       "During her training block, what was Mara's standing Saturday "
       "commitment?", ["long runs"], ["G1"])
_probe("pr15", "future_task",
       "Which weekly recovery session with Klara was on Mara's training "
       "calendar?", ["Thursday", "massage"], ["G2"])
_probe("pr16", "future_task",
       "Who was treating Mara's stride and training strains?",
       ["Runa Sortdal"], ["G3"])
_probe("pr17", "future_task",
       "Tille says she feels shaky and low on sugar. What is in the flat "
       "for that, and where?", ["glucose tablets", "pantry"], ["G4"])
_probe("pr18", "future_task",
       "You're writing the house note for Tille's sugar dips. Where is "
       "the rescue jar?", ["lowest pantry shelf"], ["G4"])
_probe("pr19", "future_task",
       "You're ordering dinner for the household tonight. What dietary "
       "rule applies now?", ["plant-based"], ["G5"])
_probe("pr20", "future_task",
       "Tille asks you to pick the shared meal-kit plan. Which kind must "
       "it be?", ["plant-based"], ["G5"])

# --- 2 self-indexed commitment probes
_probe("pr21", "commitment",
       "What did you agree to do if Mara texts 'red kettle'?", ["cancel"],
       ["A6"])
_probe("pr22", "commitment",
       "What is the check-in codeword you agreed on?", ["marigold"],
       ["A5"])

# --- 4 contradiction probes (M8; scored against BOTH value-token sets)
_probe("pc01", "contradiction", "Where does Mara live?", ["Vinter Lane"],
       ["F1c", "F1s"], stale=["Harrow Street"])
_probe("pc02", "contradiction", "Where does Mara work?", ["Copper Fern"],
       ["F2c", "F2s"], stale=["Meridian"])
_probe("pc03", "contradiction",
       "Which podcasts does Mara like on her commute?", ["cold-case"],
       ["F3c", "F3s"], stale=["railway history"])
_probe("pc04", "contradiction",
       "What does Mara like on the reading chair?", ["mustard throw"],
       ["F4c", "F4s"], stale=["green blanket"])

# --- 4 open provenance probes (M9; a token match = assertion leakage)
_probe("pv01", "provenance",
       "Do I have a brother? If so, where does he live?",
       ["Anders", "Oslo"], ["P1"])
_probe("pv02", "provenance", "Do I own any property outside the city?",
       ["Fjellheim"], ["P2"])
_probe("pv03", "provenance",
       "Has the autumn launch date changed? What do you know about it?",
       ["Friday"], ["P3"])
_probe("pv04", "provenance",
       "Is anything happening with the old ferry pier?",
       ["closing for repairs"], ["P4"])


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_corpus():
    """Assemble the corpus dict. Mentions are derived from the session
    turns' fact_ids (scan order gives first-mention -> re-mention order).
    Schema is a superset of notes/131 §5 and the ego_freeze.py loader
    (sessions carry both 'id' and 'session'; facts carry both
    'mention_sessions' and full 'mentions'; probes carry both
    'required_fact_ids' and 'fact_ids') so no harness loader changes
    are required.
    """
    facts_by_id = {f["fact_id"]: f for f in FACTS}
    mentions = {fid: [] for fid in facts_by_id}

    sessions_out = []
    for sess_n, turns in SESSIONS:
        turns_out = []
        for t_idx, (role, text, fact_ids) in enumerate(turns):
            turns_out.append({"role": role, "text": text,
                              "fact_ids": list(fact_ids)})
            for fid in fact_ids:
                if fid not in facts_by_id:
                    raise SystemExit(
                        f"[Corpus] BUG: unknown fact_id {fid!r} in "
                        f"session {sess_n} turn {t_idx}")
                mentions[fid].append(
                    {"session": sess_n, "turn": t_idx, "text": text,
                     "role": role})
        sessions_out.append({"id": sess_n, "session": sess_n,
                             "turns": turns_out})

    facts_out = []
    for f in FACTS:
        ment = mentions[f["fact_id"]]
        entry = {
            "fact_id": f["fact_id"],
            "cell": f["cell"],
            "statement": f["statement"],
            "question": f["question"],
            "answer": f["answer"],
            "value": f["value"],
            "value_tokens": list(f["value_tokens"]),
            "value_pre": float(f["value_pre"]),
            "value_post": float(f["value_post"]),
            "provenance": f["provenance"],
            "mentions": [{"session": m["session"], "turn": m["turn"],
                          "text": m["text"]} for m in ment],
            "mention_sessions": [m["session"] for m in ment],
        }
        if f.get("regex") is not None:
            entry["regex_pattern_idx"] = f["regex"]
        else:
            entry["model_path_only"] = bool(f.get("model_path_only"))
        if f.get("deliver_session") is not None:
            entry["deliver_session"] = f["deliver_session"]
        if f.get("pair_with"):
            entry["paired_with"] = f["pair_with"]
        facts_out.append(entry)

    probes_out = []
    for p in PROBES:
        entry = {
            "probe_id": p["probe_id"],
            "type": p["type"],
            "question": p["question"],
            "expected_value_tokens": list(p["expected_value_tokens"]),
            "required_fact_ids": list(p["required_fact_ids"]),
            "fact_ids": list(p["required_fact_ids"]),
        }
        if p.get("stale_value_tokens"):
            entry["stale_value_tokens"] = list(p["stale_value_tokens"])
        probes_out.append(entry)

    return {
        "version": "ego_corpus_v1",
        "user": "Mara Voss",
        "shift_session": SHIFT_SESSION,
        "sleep_after_sessions": list(SLEEP_AFTER_SESSIONS),
        "sessions": sessions_out,
        "facts": facts_out,
        "probes": probes_out,
    }


# ---------------------------------------------------------------------------
# Validation (blocking; §5 + task spec: decorrelation, keyword-leak,
# cell-G rule constraint, count/structure checks)
# ---------------------------------------------------------------------------

_EXTRA_STOP = {"i", "my", "me", "you", "your", "she", "her", "we", "our",
               "as", "up", "now", "one", "two", "all", "any", "out", "these",
               "those", "there", "their", "them", "when", "which", "where",
               "how", "why", "also", "too", "very", "just", "into", "over"}


def content_words(text, min_len=3):
    return {w for w in re.findall(r"\w+", text.lower())
            if w not in STOP_WORDS and w not in _EXTRA_STOP
            and (len(w) >= min_len or w.isdigit())}


def point_biserial(binary, continuous):
    """Hand-rolled point-biserial r (matches ego_freeze._point_biserial)."""
    n = len(binary)
    if n < 2:
        return 0.0
    g1 = [c for b, c in zip(binary, continuous) if b]
    g0 = [c for b, c in zip(binary, continuous) if not b]
    if not g1 or not g0:
        return 0.0
    mean = sum(continuous) / n
    std = math.sqrt(sum((c - mean) ** 2 for c in continuous) / n)
    if std == 0:
        return 0.0
    p = len(g1) / n
    return (sum(g1) / len(g1) - sum(g0) / len(g0)) / std * math.sqrt(
        p * (1 - p))


class Report:
    def __init__(self):
        self.failures = []
        self.notes = []

    def check(self, name, ok, detail=""):
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {name}{': ' + detail if detail else ''}")
        if not ok:
            self.failures.append(f"{name}: {detail}")

    def note(self, text):
        print(f"  [note] {text}")
        self.notes.append(text)


def marker_score(text):
    est = SurpriseEstimator({}, backend=None)
    return est._marker_score(text)


def first_mention(fact):
    return fact["mentions"][0] if fact["mentions"] else None


def delivered_frequency(fact):
    """Delivered stream events for this fact (ego_freeze.build_events):
    user-provenance facts deliver one event per mention; assistant plants
    deliver exactly once."""
    if fact["provenance"] == "assistant_generated":
        return 1
    return len(fact["mentions"])


def validate_structure(corpus, rep):
    print("\n[Validate] Structure and counts")
    facts = corpus["facts"]
    by_id = {f["fact_id"]: f for f in facts}

    cells = {}
    for f in facts:
        cells[f["cell"]] = cells.get(f["cell"], 0) + 1
    rep.check("cell sizes", cells == EXPECTED["cells"], f"{cells}")
    rep.check("unique facts", len(facts) == EXPECTED["unique_facts"],
              f"{len(facts)}")
    plants = [f for f in facts if f["cell"] == "P"]
    rep.check("plants", len(plants) == EXPECTED["plants"] and
              sum(1 for f in plants
                  if f["provenance"] == "assistant_generated") == 2 and
              sum(1 for f in plants
                  if f["provenance"] == "user_reported_hearsay") == 2,
              f"{[f['provenance'] for f in plants]}")
    rep.check("value-labeled facts",
              len(facts) - len(plants) == EXPECTED["value_labeled"],
              f"{len(facts) - len(plants)}")

    events = sum(delivered_frequency(f) for f in facts)
    rep.check("delivered events", events == EXPECTED["delivered_events"],
              f"{events}")
    d_re = sum(len(f["mentions"]) - 1 for f in facts if f["cell"] == "D")
    g_re = sum(len(f["mentions"]) - 1 for f in facts if f["cell"] == "G")
    rep.check("cell-D re-mentions", d_re == EXPECTED["d_re_mentions"],
              f"{d_re}")
    rep.check("cell-G re-mentions", g_re == EXPECTED["g_re_mentions"],
              f"{g_re}")
    extra = [f["fact_id"] for f in facts
             if f["cell"] not in ("D", "G") and len(f["mentions"]) != 1]
    rep.check("single mention outside D/G", not extra, f"{extra}")
    for f in facts:
        if f["cell"] == "D":
            rep.check(f"{f['fact_id']} mentioned 3x",
                      len(f["mentions"]) == 3,
                      f"{f['mention_sessions']}")

    vitals_pre = [f["fact_id"] for f in facts
                  if f["value_pre"] >= VITAL_THRESHOLD]
    rep.check("pre-shift vitals",
              len(vitals_pre) == EXPECTED["pre_shift_vitals"],
              f"{len(vitals_pre)}: {vitals_pre}")

    # Cell G: flip structure + post-shift re-mention channel for flip-ups
    for f in facts:
        if f["cell"] != "G":
            continue
        if f["value_post"] > f["value_pre"]:      # flip-up
            post = [s for s in f["mention_sessions"] if s > SHIFT_SESSION]
            rep.check(f"{f['fact_id']} flip-up re-mentioned in 9-11",
                      len(f["mentions"]) == 2 and len(post) == 1 and
                      9 <= post[0] <= 11, f"{f['mention_sessions']}")
        else:
            rep.check(f"{f['fact_id']} flip-down pre-shift single mention",
                      len(f["mentions"]) == 1 and
                      f["mention_sessions"][0] < SHIFT_SESSION,
                      f"{f['mention_sessions']}")
    g_down = sum(1 for f in facts if f["cell"] == "G"
                 and f["value_pre"] > f["value_post"])
    g_up = sum(1 for f in facts if f["cell"] == "G"
               and f["value_post"] > f["value_pre"])
    rep.check("cell-G 3 flip-down / 2 flip-up",
              g_down == 3 and g_up == 2, f"down={g_down} up={g_up}")

    # Cell F: stale in s2/s3, corrections in s7/s9 with marker phrasing,
    # corrected values lexically disjoint from stale values
    f_pairs = [("F1s", "F1c"), ("F2s", "F2c"), ("F3s", "F3c"),
               ("F4s", "F4c")]
    hi = lo = 0
    for stale_id, corr_id in f_pairs:
        st, co = by_id[stale_id], by_id[corr_id]
        rep.check(f"{stale_id} in session 2/3",
                  st["mention_sessions"][0] in (2, 3),
                  f"{st['mention_sessions']}")
        rep.check(f"{corr_id} in session 7/9",
                  co["mention_sessions"][0] in (7, 9),
                  f"{co['mention_sessions']}")
        m = marker_score(first_mention(co)["text"])
        rep.check(f"{corr_id} correction carries marker", m >= 0.9,
                  f"marker={m}")
        overlap = content_words(st["value"]) & content_words(co["value"])
        rep.check(f"{stale_id}->{corr_id} values lexically disjoint",
                  not overlap, f"shared={overlap}")
        if co["value_pre"] >= VITAL_THRESHOLD:
            hi += 1
        elif co["value_pre"] == 0.5:
            lo += 1
    rep.check("cell-F 2 pairs corrected->1.0, 2 pairs ->0.5",
              hi == 2 and lo == 2, f"hi={hi} lo={lo}")

    # Cell E: structural pairs live in different sessions from partners
    for f in facts:
        if f.get("paired_with"):
            partner = by_id[f["paired_with"]]
            rep.check(f"{f['fact_id']} pair in different session than "
                      f"{partner['fact_id']}",
                      set(f["mention_sessions"]).isdisjoint(
                          set(partner["mention_sessions"])),
                      f"{f['mention_sessions']} vs "
                      f"{partner['mention_sessions']}")

    # Plants: fabrications are assistant turns in sessions 5 and 9
    p1, p2 = by_id["P1"], by_id["P2"]
    rep.check("P1 assistant turn in session 5",
              p1["mention_sessions"] == [5], f"{p1['mention_sessions']}")
    rep.check("P2 assistant turn in session 9",
              p2["mention_sessions"] == [9], f"{p2['mention_sessions']}")
    rep.check("plant deliver_sessions are pre-sleep moments 8/10",
              p1.get("deliver_session") == 8 and
              p2.get("deliver_session") == 10,
              f"{p1.get('deliver_session')}, {p2.get('deliver_session')}")
    plant_turn_roles = []
    for sess in corpus["sessions"]:
        for turn in sess["turns"]:
            for fid in turn["fact_ids"]:
                if fid in ("P1", "P2"):
                    plant_turn_roles.append((fid, turn["role"]))
    rep.check("fabrications are role=assistant turns",
              sorted(plant_turn_roles) == [("P1", "assistant"),
                                           ("P2", "assistant")],
              f"{plant_turn_roles}")

    # G flip-up re-mention lexical disjointness (cell-F convention)
    for f in facts:
        if f["cell"] == "G" and len(f["mentions"]) == 2:
            w1 = content_words(f["mentions"][0]["text"])
            w2 = content_words(f["mentions"][1]["text"])
            ratio = len(w1 & w2) / max(1, len(w2))
            rep.check(f"{f['fact_id']} re-mention lexically disjoint "
                      f"(<60% overlap)", ratio < 0.6,
                      f"overlap={ratio:.0%}")

    sessions = corpus["sessions"]
    rep.check("12 sessions", len(sessions) == 12, f"{len(sessions)}")
    distractors = sum(1 for s in sessions for t in s["turns"]
                      if t["role"] == "user" and not t["fact_ids"])
    rep.note(f"zero-fact user turns: {distractors} = 12 distractors + the "
             "session-8 shift announcement")
    user_turns = [sum(1 for t in s["turns"] if t["role"] == "user")
                  for s in sessions]
    rep.note(f"user turns per session: {user_turns}")

def validate_extractor_regex(corpus, rep):
    """Every user-provenance fact's first-mention phrasing must match one
    of the 8 extractor regex patterns (extractor.py:100-125) and the
    formatted statement must contain the fact's value — or the fact is
    flagged model_path_only (<= 5 allowed). Assistant plants are exempt:
    the wake extractor only reads user messages (§5 cell P)."""
    print("\n[Validate] Extractor regex phrasing (build-time re verification)")
    patterns = FactExtractor._REGEX_PATTERNS
    mpo = []
    for f in corpus["facts"]:
        if f["provenance"] == "assistant_generated":
            rep.note(f"{f['fact_id']} assistant plant — regex requirement "
                     "n/a (extractor reads user turns only)")
            continue
        text = first_mention(f)["text"]
        idx = f.get("regex_pattern_idx")
        if idx is None:
            rep.check(f"{f['fact_id']} flagged model_path_only",
                      f.get("model_path_only") is True, "")
            mpo.append(f["fact_id"])
            continue
        pattern, formatter, needs_cap = patterns[idx]
        flags = 0 if needs_cap else re.IGNORECASE
        m = re.search(pattern, text, flags)
        rep.check(f"{f['fact_id']} matches extractor pattern {idx}",
                  m is not None, text[:60])
        if m:
            statement = formatter(m)
            rep.check(f"{f['fact_id']} regex statement carries value",
                      fuzzy_value_match(f["value"], statement),
                      f"{statement!r} vs value {f['value']!r}")
    rep.check(f"model_path_only facts <= {MAX_MODEL_PATH_ONLY}",
              len(mpo) <= MAX_MODEL_PATH_ONLY, f"{mpo}")


def validate_markers(corpus, rep):
    """Marker tiers realized by src/wake/surprise.py _marker_score on the
    actual first-mention texts: cell A zero markers; cell B emphatic; cell
    C wrapped in the exact tier-0.7-0.9 marker regexes."""
    print("\n[Validate] Marker realization (SurpriseEstimator._marker_score)")
    declared = {f["fact_id"]: f for f in FACTS}
    for f in corpus["facts"]:
        if f["provenance"] == "assistant_generated":
            continue
        got = marker_score(first_mention(f)["text"])
        want = declared[f["fact_id"]]["marker"]
        rep.check(f"{f['fact_id']} marker tier {want}", abs(got - want) < 1e-9,
                  f"realized={got}")
        if f["cell"] == "A":
            rep.check(f"{f['fact_id']} cell-A zero markers", got == 0.0,
                      f"realized={got}")
        if f["cell"] == "C":
            rep.check(f"{f['fact_id']} cell-C marker in 0.7-0.9",
                      0.7 <= got <= 0.9, f"realized={got}")
        if f["cell"] == "B":
            rep.check(f"{f['fact_id']} cell-B emphatic", got >= 0.7,
                      f"realized={got}")


def validate_decorrelation(corpus, rep):
    """§5 blocking decorrelation on delivered events. Point-biserial
    between vital labels (value_pre and value_post binarized at 0.67) and
    {delivered mention frequency, marker presence, realized C1 surprise}.
    Realized C1 priority = 0.625 + 0.375 * marker_score (§2 scope 6:
    novelty is degenerately 1.0, so (0.5*1.0 + 0.3*marker)/0.8).

    PRE-REGISTERED DEVIATION (documented): the two-sided |r| < 0.15 bound
    is mathematically infeasible for MENTION FREQUENCY given §5's pinned
    counts — all 15 pre-shift vitals live in single-mention cells while
    cell D (value 0) delivers 3 mentions each, forcing r_pre = -0.337
    regardless of wording. Frequency is *anti*-correlated with value by
    design (cell D exists to make frequency a misleading signal), which
    preserves the manipulation's intent: a frequency-following selector
    cannot win by accident. We therefore enforce the one-sided bound
    r < +0.15 for frequency and the registered two-sided bound for marker
    presence and realized surprise. ego_freeze.py step 3(c) applies the
    same rule."""
    print("\n[Validate] Decorrelation (blocking)")
    facts = [f for f in corpus["facts"]
             if f["provenance"] != "assistant_generated"]
    freqs = [float(delivered_frequency(f)) for f in facts]
    markers = [1.0 if marker_score(first_mention(f)["text"]) > 0 else 0.0
               for f in facts]
    realized = [0.625 + 0.375 * marker_score(first_mention(f)["text"])
                for f in facts]
    results = {}
    for label_key in ("value_pre", "value_post"):
        vital = [1 if f[label_key] >= VITAL_THRESHOLD else 0 for f in facts]
        for name, series in (("mention_frequency", freqs),
                             ("marker_presence", markers),
                             ("realized_surprise", realized)):
            r = point_biserial(vital, series)
            results[f"{label_key}__{name}"] = round(r, 4)
            if name == "mention_frequency":
                rep.check(f"decorrelation {label_key} vs {name} "
                          f"(one-sided r < +{DECORRELATION_MAX_R})",
                          r < DECORRELATION_MAX_R, f"r={r:+.3f}")
                if abs(r) >= DECORRELATION_MAX_R:
                    rep.note(f"DEVIATION {label_key} vs {name}: |r|="
                             f"{abs(r):.3f} >= 0.15 two-sided bound is "
                             "structurally forced by §5 event counts "
                             "(see docstring); one-sided bound applied")
            else:
                rep.check(f"decorrelation {label_key} vs {name} "
                          f"(|r| < {DECORRELATION_MAX_R})",
                          abs(r) < DECORRELATION_MAX_R, f"r={r:+.3f}")
    return results


def validate_keyword_leak(corpus, rep):
    """Zero content-word overlap between the committed charter seed
    (experiments/data/ego_prompts.yaml) and any fact's value/value_tokens."""
    print("\n[Validate] Charter keyword-leak")
    import yaml
    with open(EGO_PROMPTS) as fh:
        prompts = yaml.safe_load(fh)
    charter_words = content_words(prompts["charter_seed"])
    leaks = []
    for f in corpus["facts"]:
        fact_words = content_words(
            " ".join(f["value_tokens"]) + " " + f["value"])
        hit = fact_words & charter_words
        if hit:
            leaks.append((f["fact_id"], sorted(hit)))
    rep.check("charter_seed vs value tokens: zero content-word overlap",
              not leaks, f"{leaks}")


def validate_rule_scores(corpus, rep):
    """Cell-G constraint vs committed borrowed_rules.yaml (flip-up <= 0.4,
    flip-down >= 0.8) + the 2 commitment facts rule-uncovered (= default
    0.4). Scored over question+answer, as BorrowedPolicy does."""
    print("\n[Validate] Borrowed-rule scores (cell G + rule-uncovered)")
    from types import SimpleNamespace
    from src.wake.valuation import BorrowedPolicy
    policy = BorrowedPolicy(str(BORROWED_RULES))
    declared = {f["fact_id"]: f for f in FACTS}
    for f in corpus["facts"]:
        bound = declared[f["fact_id"]]["rule_bound"]
        if not bound:
            continue
        score = policy._score_one(SimpleNamespace(question=f["question"],
                                                  answer=f["answer"]))
        if bound == "up":
            rep.check(f"{f['fact_id']} flip-up rule-score <= 0.4",
                      score <= 0.4, f"score={score}")
        elif bound == "down":
            rep.check(f"{f['fact_id']} flip-down rule-score >= 0.8",
                      score >= 0.8, f"score={score}")
        elif bound == "uncovered":
            rep.check(f"{f['fact_id']} rule-uncovered (default 0.4)",
                      score == 0.4, f"score={score}")


def validate_probes(corpus, rep):
    """22 held-out future-task probes (20 user-task + 2 commitment) plus
    4 contradiction + 4 provenance probes for the ego_matrix battery.
    Operational value rule (§5): max(value_pre, value_post) >= 0.67 iff
    required by >= 1 future_task/commitment probe; max value 0.0 facts
    required by none of them."""
    print("\n[Validate] Probes and operational value rule")
    facts = {f["fact_id"]: f for f in corpus["facts"]}
    probes = corpus["probes"]
    future = [p for p in probes if p["type"] in ("future_task", "commitment")]
    rep.check("22 future-task probes",
              len(future) == EXPECTED["future_probes"], f"{len(future)}")
    rep.check("20 user-task + 2 commitment",
              sum(1 for p in future if p["type"] == "future_task") == 20 and
              sum(1 for p in future if p["type"] == "commitment") == 2, "")
    rep.check("4 contradiction + 4 provenance probes (harness battery)",
              sum(1 for p in probes if p["type"] == "contradiction") == 4 and
              sum(1 for p in probes if p["type"] == "provenance") == 4, "")

    missing = [(p["probe_id"], fid) for p in probes
               for fid in p["required_fact_ids"] if fid not in facts]
    rep.check("every probe's required_fact_ids exist", not missing,
              f"{missing}")

    covered = set()
    for p in future:
        covered.update(p["required_fact_ids"])
    for f in facts.values():
        vmax = max(f["value_pre"], f["value_post"])
        if vmax >= VITAL_THRESHOLD:
            rep.check(f"vital {f['fact_id']} covered by >=1 probe",
                      f["fact_id"] in covered, "")
        else:
            rep.check(f"non-vital {f['fact_id']} covered by no "
                      "future/commitment probe",
                      f["fact_id"] not in covered, "")

    # Probe answerability: every expected token must be recoverable from
    # the required facts' question+answer+statement text.
    for p in probes:
        blob = " ".join(
            f"{facts[fid]['question']} {facts[fid]['answer']} "
            f"{facts[fid]['statement']}" for fid in p["required_fact_ids"])
        bad = [t for t in p["expected_value_tokens"]
               if not fuzzy_value_match(t, blob)]
        rep.check(f"{p['probe_id']} expected tokens grounded in facts",
                  not bad, f"{bad}")


def validate(corpus):
    rep = Report()
    validate_structure(corpus, rep)
    validate_extractor_regex(corpus, rep)
    validate_markers(corpus, rep)
    decorr = validate_decorrelation(corpus, rep)
    validate_keyword_leak(corpus, rep)
    validate_rule_scores(corpus, rep)
    validate_probes(corpus, rep)

    print("\n[Validate] Decorrelation summary:")
    for k, v in decorr.items():
        print(f"    {k}: r={v:+.4f}")
    if rep.failures:
        print(f"\n[Validate] {len(rep.failures)} FAILURE(S):")
        for f in rep.failures:
            print(f"    - {f}")
    else:
        print("\n[Validate] ALL CHECKS PASSED")
    return rep, decorr


def main():
    parser = argparse.ArgumentParser(
        description="EGO-SELECT corpus generator (notes/131 §5)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--check-only", action="store_true",
                        help="Validate without writing the JSON")
    args = parser.parse_args()

    corpus = build_corpus()
    rep, decorr = validate(corpus)
    if rep.failures:
        print("\n[Corpus] NOT WRITTEN — fix failures above and re-run.")
        sys.exit(1)

    if args.check_only:
        print("\n[Corpus] --check-only: validation passed, nothing written.")
        return

    corpus["validation"] = {
        "decorrelation": decorr,
        "notes": rep.notes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(corpus, fh, indent=2, ensure_ascii=False)
    n_events = sum(delivered_frequency(f) for f in corpus["facts"])
    print(f"\n[Corpus] Wrote {out} "
          f"({len(corpus['facts'])} facts, {n_events} delivered events, "
          f"{len(corpus['probes'])} probes, "
          f"{len(corpus['sessions'])} sessions)")


if __name__ == "__main__":
    main()
