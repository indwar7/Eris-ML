"""Build the raw corpus for "Digitization Pipeline Attribution": real historical
newspaper OCR text, grouped by the real digitization BATCH it came from.

Source: Chronicling America / National Digital Newspaper Program (NDNP), a
Library of Congress program, via the plain static-file bulk mirror

    https://chroniclingamerica.loc.gov/data/batches/

(the modern chroniclingamerica.loc.gov *API* and loc.gov endpoints sit behind
Cloudflare and 403 for programmatic access; this bulk-data file index does
not).  Layout on that mirror:

    batches/<batch_id>/data/batch.xml                       -- manifest: every
                                                                 issue in the
                                                                 batch, as
                                                                 (lccn, date,
                                                                 path to the
                                                                 issue's METS)
    batches/<batch_id>/data/<lccn>/<reel>/<issue>/<issue>.xml -- issue-level
                                                                 METS, lists
                                                                 the OCR ALTO
                                                                 file for each
                                                                 page
    .../<page>.xml                                            -- page-level
                                                                 ALTO 2.0 XML:
                                                                 real OCR
                                                                 vendor
                                                                 metadata
                                                                 (<OCRProcessing>)
                                                                 plus the OCR
                                                                 text itself,
                                                                 one
                                                                 <TextBlock>
                                                                 per
                                                                 article/column
                                                                 region, each
                                                                 built from
                                                                 <String
                                                                 CONTENT="...">
                                                                 tokens.

A batch is a single NDNP award: one contributing institution digitized a
run of one or more titles through one (real, contracted) OCR vendor
pipeline.  Different batches typically used different vendors/pipeline
versions, which leaves a real, measurable character-error fingerprint in the
raw OCR text -- confirmed in a prior 4-batch gate-check (chance 0.250, best
single cheap scalar 0.407, full noise-feature logistic-regression model
0.742).  This script only fetches the raw material; it does not design the
challenge task, metric, splits or bags.

Run:  python build_raw.py [out_dir]

Streams to disk as it goes and is safe to re-run: batches already present in
batches_meta.csv are skipped.  NOTE: this is a *live* archive fetch, not a
byte-identical-reproducible generator -- re-running it later will not
necessarily reproduce exactly the same snippets (pages can be re-processed,
mirror layout can shift).  The committed dataset/raw/ files are the frozen
source of truth for the challenge; this script documents/replicates the
fetch, in the same spirit as this project's other real-data challenges.
"""
from pathlib import Path
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

BASE = "https://chroniclingamerica.loc.gov/data/batches/"
UA = {"User-Agent": "eris-benchmark-build/1.0 (research; contact via Shipd)"}

REQ_PAUSE = 1.1           # be a good citizen with the LOC mirror -- measured empirically:
                          # 0.45s triggered frequent, expensive 429 backoffs; this rate
                          # (~55 req/min) ran clean in practice.
RATE_LIMIT_BACKOFF = 30   # seconds to sleep after a 429 before resuming
MAX_RETRIES = 3
FETCH_TIMEOUT = 25        # a legitimate response (batch.xml, issue METS, one ALTO
                          # page) is at most ~1MB; anything hung this long is dead

DATES_PER_BATCH = 18      # distinct issue dates sampled per batch (spread across the batch's run)
BLOCKS_PER_PAGE_CAP = 14  # cap on TextBlock snippets kept from a single page
MIN_BLOCK_CHARS = 40      # drop TextBlocks whose concatenated OCR text is this short
MIN_DATES_TO_KEEP = 4     # skip a batch whose batch.xml lists fewer distinct dates than this
MIN_SNIPPETS_TO_KEEP = 60 # skip a batch that ends up this thin after fetching
TARGET_BATCHES = 90       # stop once this many good batches are collected
MAX_TIME_SECONDS = 60 * 9  # soft wall-clock budget for THIS invocation (script is
                           # re-invoked repeatedly across several tool calls to reach
                           # TARGET_BATCHES; kept short so each call finishes cleanly
                           # inside a single tool-call timeout instead of getting killed
                           # mid-request).

# 319 candidate batch IDs -- every prefix present in the mirror's batch index,
# 6 batches per prefix, alphabetically first-within-prefix (all verified to
# exist in the live batches/ index at fetch time; unlike an earlier draft of
# this list, these are not guessed by pattern). The 4 batches already used in
# the prior gate-check (ak_albatross_ver01, ct_aetna_ver02, mnhi_allemande_ver01,
# txdn_albatross_ver01) are included via this same sweep. We work down this
# list until TARGET_BATCHES good batches are found or the list/time budget
# runs out -- not every candidate pans out (a few have very few issues online,
# a few 404 despite being listed, a few get skipped for thinness).
CANDIDATE_BATCHES = [
    "ak_albatross_ver01", "arhi_aerosmith_ver01", "au_abernethy_ver01", "az_acacia_ver01",
    "cohi_abbeyville_ver01", "ct_aetna_ver02", "curiv_ahwahnee_ver01", "deu_accio_ver01",
    "dlc_1arp_ver01", "fu_albert_ver01", "gu_abby_ver02", "hihouml_angel_ver01",
    "iahi_abra_ver01", "idhi_angelou_ver01", "in_abraham_ver02", "iune_albatross_ver01",
    "khi_allen_ver02", "kyu_airplane_ver01", "lu_angel_ver01", "mb_allspice_ver01",
    "mdu_alvarez_ver01", "me_acadien_ver01", "mimtptc_adrian_ver01", "mnhi_allemande_ver01",
    "mohi_angelou_ver03", "msar_abolitionist_ver02", "mthi_adderstongue_ver01", "nbu_abbott_ver01",
    "ncu_adam_ver02", "ndhi_alamo_ver01", "nhd_avalon_ver02", "njr_allspice_ver02",
    "nmu_agave_ver01", "nn_absaber_ver01", "nvln_arrowhead_ver01", "ohi_alastor_ver02",
    "okhi_apache_ver02", "oru_argonaut_ver01", "prru_abeja_ver01", "pst_altoona_ver02",
    "rp_aboleth_ver01", "scu_albinoskunk_ver02", "sdhi_apple_ver01", "tu_anita_ver01",
    "txdn_albatross_ver01", "uriv_elodia_ver01", "uuml_anderson_ver01", "vi_abingdon_ver01",
    "vnstcsc_adelphi_ver01", "vtu_adamant_ver01", "wa_alder_ver01", "whi_ada_ver01",
    "wvu_antares_ver02", "wyu_aarakocra_ver01", "ak_amaranth_ver01", "arhi_alakazam_ver01",
    "au_andrews_ver01", "az_agave_ver01", "cohi_aberdeen_ver01", "ct_algaesoup_ver02",
    "curiv_albion_ver01", "deu_arden_ver01", "dlc_1bernal_ver01", "fu_anastasia_ver01",
    "gu_ace_ver02", "hihouml_ariel_ver01", "iahi_aerodactyl_ver01", "idhi_angkor_ver01",
    "in_alford_ver01", "iune_alpha_ver01", "khi_anthony_ver01", "kyu_albatross_ver01",
    "lu_arbok_ver04", "mb_artemis_ver01", "mdu_anhinga_ver01", "me_allagash_ver02",
    "mimtptc_albion_ver01", "mnhi_amboy_ver02", "mohi_ansel_ver01", "msar_agate_ver01",
    "mthi_alderfly_ver01", "nbu_alliance_ver01", "ncu_alligator_ver01", "ndhi_almont_ver01",
    "nhd_avocet_ver01", "njr_anthony_ver01", "nmu_antelope_ver01", "nn_angelou_ver01",
    "nvln_aurora_ver02", "ohi_alpha_ver01", "okhi_atoka_ver01", "oru_ashland_ver01",
    "prru_abelardo_ver01", "pst_armsby_ver01", "rp_annishag_ver04", "scu_alexiavalentine_ver01",
    "sdhi_aruba_ver01", "tu_archie_ver01", "txdn_alpha_ver01", "uuml_armstrong_ver01",
    "vi_adams_ver01", "vnstcsc_annaly_ver02", "vtu_alburg_ver02", "wa_alki_ver01",
    "whi_altbrew_ver01", "wvu_archer_ver01", "wyu_afton_ver01", "ak_arcticfox_ver02",
    "arhi_andromeda_ver01", "au_ayler_ver01", "az_anthillgarnet_ver04", "cohi_alta_ver01",
    "ct_andover_ver01", "curiv_angelica_ver01", "deu_artemis_ver02", "dlc_1chagall_ver01",
    "fu_anderson_ver02", "gu_alicorn_ver02", "hihouml_azure_ver01", "iahi_allison_ver01",
    "idhi_atkinson_ver02", "in_archer_ver01", "iune_amethyst_ver01", "khi_arbuckle_ver02",
    "kyu_aluminum_ver01", "lu_arminius_ver01", "mb_basil_ver01", "mdu_annapolis_ver02",
    "me_aroostook_ver01", "mimtptc_alma_ver01", "mnhi_angus_ver02", "mohi_archie_ver01",
    "msar_applejack_ver01", "mthi_anaconda_ver01", "nbu_americanrobin_ver01", "ncu_apple_ver01",
    "ndhi_andorian_ver01", "nhd_bondcliff_ver01", "njr_asburypark_ver01", "nmu_austen_ver01",
    "nn_aristotle_ver01", "nvln_beatty_ver02", "ohi_amaryllis_ver01", "okhi_avocado_ver01",
    "oru_auklet_ver01", "prru_aguada_ver01", "pst_atherton_ver01", "rp_astralstalker_ver01",
    "scu_andersonpink_ver02", "sdhi_avenger_ver01", "tu_arthur_ver01", "txdn_andrews_ver01",
    "uuml_basso_ver01", "vi_adams_ver02", "vnstcsc_belvedere_ver03", "vtu_asparagus_ver01",
    "wa_american_ver02", "whi_arbutus_ver01", "wvu_armstrong_ver02", "wyu_alcott_ver01",
    "ak_arctictern_ver01", "arhi_aragonite_ver01", "au_baskin_ver01", "az_apachetrout_ver01",
    "cohi_bailey_ver02", "ct_animals_ver01", "curiv_angwin_ver02", "deu_batman_ver01",
    "dlc_1duchamp_ver01", "fu_arcadia_ver02", "gu_ara_ver01", "hihouml_brick_ver01",
    "iahi_ames_ver01", "idhi_bagrati_ver02", "in_ashbel_ver01", "iune_archives_ver01",
    "khi_bender_ver01", "kyu_aussie_ver01", "lu_beast_ver01", "mb_bia_ver01",
    "mdu_aster_ver02", "me_ashdale_ver01", "mimtptc_alpena_ver01", "mnhi_anoka_ver02",
    "mohi_beetlebailey_ver01", "msar_bamboo_ver01", "mthi_antelope_ver02", "nbu_ancientbison_ver01",
    "ncu_appling_ver02", "ndhi_argon_ver01", "nhd_cardinal_ver01", "njr_bacala_ver01",
    "nmu_barberry_ver04", "nn_bentham_ver01", "nvln_bullfrog_ver01", "ohi_ariel_ver02",
    "okhi_beaver_ver01", "oru_belknap_ver01", "prru_arroz_ver02", "pst_beaver_ver01",
    "rp_azer_ver02", "scu_asparagus_ver01", "sdhi_banana_ver02", "tu_bertha_ver01",
    "txdn_argentina_ver01", "uuml_boozer_ver01", "vi_aerosmith_ver01", "vnstcsc_bugbyhole_ver01",
    "vtu_barre_ver01", "wa_auklet_ver01", "whi_asiago_ver02", "wvu_astor_ver01",
    "wyu_baggs_ver01", "ak_belugawhale_ver01", "arhi_beatles_ver01", "au_bolden_ver01",
    "az_arguingmatch_ver02", "cohi_baldwin_ver01", "ct_arnold_ver02", "curiv_archylee_ver01",
    "deu_bear_ver01", "dlc_1ernst_ver01", "fu_armadillo_ver01", "gu_beaker_ver01",
    "hihouml_brutus_ver02", "iahi_beiderbecke_ver01", "idhi_baldacci_ver02", "in_asimov_ver02",
    "iune_article_ver01", "khi_brockovich_ver02", "kyu_basenji_ver01", "lu_blastoise_ver01",
    "mb_circe_ver01", "mdu_avocado_ver01", "me_bangor_ver02", "mimtptc_baldwin_ver01",
    "mnhi_antares_ver03", "mohi_berenice_ver01", "msar_beryl_ver01", "mthi_avocet_ver01",
    "nbu_arwen_ver01", "ncu_black_ver02", "ndhi_bajoran_ver01", "nhd_carrigain_ver01",
    "njr_basil_ver02", "nmu_beaver_ver01", "nn_borges_ver01", "nvln_caliente_ver02",
    "ohi_arnarson_ver01", "okhi_bokchito_ver01", "oru_belladonna_ver01", "prru_bacalao_ver04",
    "pst_berks_ver01", "rp_barbeddevil_ver02", "scu_babytate_ver01", "sdhi_bearcat_ver02",
    "tu_bonnielou_ver01", "txdn_ash_ver01", "uuml_collins_ver01", "vi_affirmed_ver01",
    "vnstcsc_canaan_ver01", "vtu_broccoli_ver01", "wa_bainbridge_ver01", "whi_augurey_ver01",
    "wvu_atkinson_ver01", "wyu_beholder_ver01", "ak_bluewhale_ver01", "arhi_biotite_ver01",
    "au_brown_ver01", "az_bentonite_ver02", "cohi_bowerman_ver01", "ct_ash_ver01",
    "curiv_benicia_ver01", "deu_bombarda_ver01", "dlc_1freud_ver01", "fu_astor_ver04",
    "gu_bigfoot_ver02", "hihouml_butterfly_ver01", "iahi_bellsprout_ver01", "idhi_bronte_ver01",
    "in_bashir_ver02", "iune_azalea_ver02", "khi_brown_ver01", "kyu_batman_ver01",
    "lu_bolivar_ver01", "mb_demeter_ver01", "mdu_bean_ver01", "me_baxter_ver01",
    "mimtptc_bath_ver01", "mnhi_antony_ver01", "mohi_boone_ver01", "msar_braeburn_ver01",
    "mthi_beargrass_ver01", "nbu_azureaster_ver01", "ncu_blueberry_ver01", "ndhi_beryllium_ver01",
    "nhd_doublehead_ver02", "njr_beachhaven_ver01", "nmu_bronte_ver01", "nn_brown_ver01",
    "nvln_carlin_ver01", "ohi_atticus_ver01", "okhi_bologna_ver02", "oru_bobolink_ver01",
    "prru_ballena_ver01", "pst_borland_ver01", "rp_beholder_ver02", "scu_blueberries_ver01",
    "sdhi_bermuda_ver01", "tu_brownie_ver01", "txdn_aubrey_ver01", "uuml_contador_ver01",
    "vi_albion_ver01", "vnstcsc_christiansted_ver02", "vtu_burlington_ver01", "wa_birch_ver01",
    "whi_belle_ver01", "wvu_austria_ver01", "wyu_bradbury_ver01",
]


# --------------------------------------------------------------------------- io

def fetch(url):
    last = None
    for a in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as fh:
                return fh.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"    429 rate-limited on {url}; backing off "
                      f"{RATE_LIMIT_BACKOFF}s", flush=True)
                time.sleep(RATE_LIMIT_BACKOFF)
                last = e
                continue
            if e.code in (404, 403):
                raise
            last = e
            time.sleep(2 ** a)
        except Exception as e:                                   # noqa: BLE001
            last = e
            time.sleep(2 ** a)
    raise last if last else RuntimeError(url)


def get(url):
    time.sleep(REQ_PAUSE)
    return fetch(url)


# ----------------------------------------------------------------- xml helpers

def local(tag):
    return tag.rsplit("}", 1)[-1]


def strip_ns_findall(elem, name):
    return [c for c in elem.iter() if local(c.tag) == name]


# -------------------------------------------------------------------- batch.xml

def list_batch_issues(batch_id):
    """[(lccn, issueDate, mets_relpath)] straight off batch.xml.

    Parsed with ElementTree (namespace-stripped), not a fixed-attribute-order
    regex: different awardees' batch.xml files were generated by different
    NDNP tooling versions and disagree on attribute order (e.g. `lccn=...
    issueDate=...` vs `editionOrder=... issueDate=... lccn=...`) and on
    xmlns declaration -- a regex tied to one order silently returned zero
    issues for the other layout.
    """
    url = BASE + batch_id + "/data/batch.xml"
    blob = get(url)
    try:
        root = ET.fromstring(blob)
    except ET.ParseError:
        return []
    out = []
    for el in strip_ns_findall(root, "issue"):
        lccn = el.attrib.get("lccn")
        date = el.attrib.get("issueDate")
        path = (el.text or "").strip()
        if lccn and date and path:
            out.append((lccn, date, path))
    return out


def sample_dates(issues, n):
    """Evenly-spaced sample of up to n distinct issue dates, one issue per date."""
    by_date = {}
    for lccn, date, path in issues:
        by_date.setdefault(date, (lccn, date, path))
    dates = sorted(by_date)
    if len(dates) <= n:
        return [by_date[d] for d in dates]
    idx = sorted(set(round(i * (len(dates) - 1) / (n - 1)) for i in range(n)))
    return [by_date[dates[i]] for i in idx]


# -------------------------------------------------------------------- issue METS

def ocr_page_paths(batch_id, mets_relpath):
    """Full URLs of the ALTO OCR files listed in one issue's METS, in page order."""
    rel = mets_relpath[2:] if mets_relpath.startswith("./") else mets_relpath
    issue_dir = "/".join(rel.split("/")[:-1])
    url = BASE + batch_id + "/data/" + rel
    blob = get(url)
    try:
        root = ET.fromstring(blob)
    except ET.ParseError:
        return []
    out = []
    for f in strip_ns_findall(root, "file"):
        if f.attrib.get("USE") != "ocr":
            continue
        for loc in strip_ns_findall(f, "FLocat"):
            href = next((v for k, v in loc.attrib.items() if k.endswith("href")), None)
            if href:
                href = href[2:] if href.startswith("./") else href
                out.append(BASE + batch_id + "/data/" + issue_dir + "/" + href)
    return out


# --------------------------------------------------------------------- ALTO page

def parse_alto_page(blob):
    """Return (software_creator, software_name, processing_agency, [block_text, ...])."""
    try:
        root = ET.fromstring(blob)
    except ET.ParseError:
        return None, None, None, []

    software_creator = software_name = processing_agency = None
    for step in strip_ns_findall(root, "ocrProcessingStep"):
        ag = strip_ns_findall(step, "processingAgency")
        if ag and ag[0].text:
            processing_agency = ag[0].text.strip()
        sc = strip_ns_findall(step, "softwareCreator")
        if sc and sc[0].text:
            software_creator = sc[0].text.strip()
        sn = strip_ns_findall(step, "softwareName")
        if sn and sn[0].text:
            software_name = sn[0].text.strip()
        break  # first ocrProcessingStep is the actual OCR engine, not post-processing

    blocks = []
    for tb in strip_ns_findall(root, "TextBlock"):
        words = [s.attrib.get("CONTENT", "") for s in strip_ns_findall(tb, "String")]
        words = [w for w in words if w]
        text = " ".join(words)
        if len(text) >= MIN_BLOCK_CHARS:
            blocks.append(text)
    return software_creator, software_name, processing_agency, blocks


# ------------------------------------------------------------------------ build

def fetch_batch(batch_id, snippets_fh, seq_counter):
    """Fetch one batch's snippets and, only if the batch is kept, write them all
    at once. Records are buffered in memory per-batch (not streamed line-by-line
    as they're produced) specifically so a mid-batch interruption (kill, crash,
    rate-limit abort) can never leave partial/orphaned snippets for a batch that
    never made it into batches_meta.csv -- the two files must stay in lockstep
    since a re-run skips any batch_id already present in batches_meta.csv."""
    try:
        issues = list_batch_issues(batch_id)
    except urllib.error.HTTPError as e:
        return {"batch_id": batch_id, "skip_reason": f"http {e.code} on batch.xml"}
    except Exception as e:                                        # noqa: BLE001
        return {"batch_id": batch_id, "skip_reason": f"error on batch.xml: {e}"}

    if not issues:
        return {"batch_id": batch_id, "skip_reason": "no issues listed"}
    n_distinct_dates_total = len({d for _, d, _ in issues})
    if n_distinct_dates_total < MIN_DATES_TO_KEEP:
        return {"batch_id": batch_id,
                "skip_reason": f"only {n_distinct_dates_total} distinct issue dates"}

    sampled = sample_dates(issues, DATES_PER_BATCH)

    prefix = batch_id.split("_")[0]
    n_snippets = 0
    dates_used = set()
    creators, names, agencies = set(), set(), set()
    buffered = []

    for lccn, date, mets_relpath in sampled:
        try:
            pages = ocr_page_paths(batch_id, mets_relpath)
        except urllib.error.HTTPError:
            continue
        except Exception as e:                                    # noqa: BLE001
            print(f"    {batch_id} {date}: issue METS error {e}", flush=True)
            continue
        if not pages:
            continue
        try:
            blob = get(pages[0])
        except urllib.error.HTTPError:
            continue
        except Exception as e:                                    # noqa: BLE001
            print(f"    {batch_id} {date}: page fetch error {e}", flush=True)
            continue

        sc, sn, ag, blocks = parse_alto_page(blob)
        if sc:
            creators.add(sc)
        if sn:
            names.add(sn)
        if ag:
            agencies.add(ag)
        if not blocks:
            continue

        dates_used.add(date)
        for bi, text in enumerate(blocks[:BLOCKS_PER_PAGE_CAP]):
            seq_counter[0] += 1
            rec = {
                "snippet_id": f"{batch_id}_{lccn}_{date}_{seq_counter[0]}_{bi}",
                "batch_id": batch_id,
                "lccn": lccn,
                "issue_date": date,
                "text": text,
            }
            buffered.append(rec)
            n_snippets += 1

    if n_snippets < MIN_SNIPPETS_TO_KEEP:
        return {"batch_id": batch_id,
                "skip_reason": f"only {n_snippets} snippets after fetch "
                                f"(sampled {len(sampled)} dates)"}

    for rec in buffered:
        snippets_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    snippets_fh.flush()
    return {
        "batch_id": batch_id,
        "institution_prefix": prefix,
        "n_snippets": n_snippets,
        "n_distinct_issue_dates": len(dates_used),
        "software_creator": "; ".join(sorted(creators)) if creators else "",
        "software_name": "; ".join(sorted(names)) if names else "",
        "processing_agency": "; ".join(sorted(agencies)) if agencies else "not_recorded",
    }


def build(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    snippets_path = out_dir / "snippets.jsonl"
    meta_csv_path = out_dir / "batches_meta.csv"
    meta_json_path = out_dir / "meta.json"

    done_batches = set()
    if meta_csv_path.exists():
        with open(meta_csv_path, encoding="utf8") as fh:
            for row in csv.DictReader(fh):
                done_batches.add(row["batch_id"])
    print(f"resuming with {len(done_batches)} batches already on disk", flush=True)

    # Persistent skip log across invocations: this script is re-run many times in
    # separate short sessions to reach a growing TARGET_BATCHES. Without this, every
    # re-run would re-fetch batch.xml for every previously-404'd/thin candidate from
    # scratch (wasteful and slow under the mirror's rate limit). Skips live in
    # meta.json under "skipped_batches" ({batch_id: reason}), merged and rewritten
    # at the end of each run.
    prior_skipped = {}
    if meta_json_path.exists():
        prev = json.loads(meta_json_path.read_text()).get("skipped_batches") or {}
        if isinstance(prev, list):  # older layout: list of {batch_id, skip_reason}
            prev = {r["batch_id"]: r["skip_reason"] for r in prev}
        prior_skipped.update(prev)
    print(f"{len(prior_skipped)} previously-skipped batch_ids will not be retried", flush=True)

    kept, skipped = [], []
    seq_counter = [0]
    start = time.time()

    csv_is_new = not meta_csv_path.exists()
    snippets_fh = open(snippets_path, "a", encoding="utf8")
    meta_fh = open(meta_csv_path, "a", newline="", encoding="utf8")
    writer = csv.writer(meta_fh)
    if csv_is_new:
        writer.writerow(["batch_id", "institution_prefix", "n_snippets",
                          "n_distinct_issue_dates", "software_creator",
                          "software_name", "processing_agency"])

    for i, batch_id in enumerate(CANDIDATE_BATCHES):
        if len(kept) >= TARGET_BATCHES:
            print(f"reached target of {TARGET_BATCHES} batches, stopping", flush=True)
            break
        if time.time() - start > MAX_TIME_SECONDS:
            print("hit time budget, stopping", flush=True)
            break
        if batch_id in done_batches or batch_id in prior_skipped:
            continue

        print(f"[{i+1}/{len(CANDIDATE_BATCHES)}] {batch_id} "
              f"(kept={len(kept)} elapsed={time.time()-start:.0f}s)", flush=True)
        result = fetch_batch(batch_id, snippets_fh, seq_counter)
        if result is None:
            continue
        if "skip_reason" in result:
            print(f"    skip: {result['skip_reason']}", flush=True)
            skipped.append(result)
            continue
        kept.append(result)
        done_batches.add(batch_id)
        writer.writerow([result["batch_id"], result["institution_prefix"],
                          result["n_snippets"], result["n_distinct_issue_dates"],
                          result["software_creator"], result["software_name"],
                          result["processing_agency"]])
        meta_fh.flush()
        print(f"    kept: {result['n_snippets']} snippets, "
              f"{result['n_distinct_issue_dates']} dates", flush=True)

    snippets_fh.close()
    meta_fh.close()

    total_snippets = seq_counter[0]
    meta = {
        "source": "Chronicling America / National Digital Newspaper Program (NDNP), "
                  "Library of Congress",
        "source_url": "https://chroniclingamerica.loc.gov/data/batches/",
        "fetch_date": time.strftime("%Y-%m-%d"),
        "fetch_approach": (
            "Static bulk-data mirror (not the Cloudflare-fronted API). For each "
            "candidate batch_id: fetch batch.xml (full issue manifest), evenly "
            f"sample up to {DATES_PER_BATCH} distinct issue dates across the "
            "batch's run, fetch each sampled issue's METS to find its first "
            "page's ALTO OCR file, fetch that page, and keep every TextBlock "
            f"with >= {MIN_BLOCK_CHARS} characters of OCR text (capped at "
            f"{BLOCKS_PER_PAGE_CAP} TextBlocks/page) as one snippet. Batches "
            f"with < {MIN_DATES_TO_KEEP} distinct issue dates in their manifest, "
            f"or that yielded < {MIN_SNIPPETS_TO_KEEP} snippets after fetching, "
            "were skipped."),
        "n_batches_kept": len(kept),
        "n_batches_skipped": len(skipped),
        "n_batches_attempted": len(kept) + len(skipped),
        "total_snippets": total_snippets,
        "wall_clock_seconds": round(time.time() - start, 1),
        "skipped_batches": {**prior_skipped,
                            **{r["batch_id"]: r["skip_reason"] for r in skipped}},
        "rate_limit_note": (
            f"On a 429 the fetcher sleeps {RATE_LIMIT_BACKOFF}s and resumes the "
            "same request rather than hammering through it."),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "skipped_batches"}, indent=2))


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "raw")
