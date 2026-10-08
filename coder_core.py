# ---------- job-title normalisation (same rules used when the reference codes were made) ----------
GENDER = re.compile(r'\(\s*(?:[mwfdxh](?:\s*[\/|,]\s*[mwfdxh])*)\s*\)', re.I)
NOTE = re.compile(r'(\d|%|\bweek\b|\bhour|\bhrs\b|\bpart[- ]?time\b|\bfull[- ]?time\b|\bwage|\bsalary|'
                  r'\bper month\b|\bimmediate|\bstart\b|\btemporary\b|\bpermanent\b|\bshift\b|'
                  r'\bapprox|\bfrom\b|\bunbefristet\b|\bvollzeit\b|\bteilzeit\b)', re.I)
NOISE = re.compile(r'\b(full[- ]?time|part[- ]?time|fulltime|parttime|permanent position|temporary|'
                   r'immediate start|urgently|wanted|sought|we are looking for|vacancy|all genders|'
                   r'm/w/d|m/f/d|w/m/d|f/m/d|m/w/x|incl|approx)\b', re.I)
STOP = {'a','an','the','of','for','in','at','on','with','and','or','to','as','by','from','our','your',
        'per','etc','is','are','we','you','new','other'}
SYN = {'salesperson':'sales assistant','saleswoman':'sales assistant','salesman':'sales assistant',
       'sale':'sales','waitress':'waiter','charwoman':'cleaner','cleaning':'cleaner','lorry':'truck',
       'lkw':'truck','hgv':'truck','it':'information technology','ict':'information communication technology',
       'hr':'human resource','qa':'quality assurance','mechatronics':'mechatronic','physician':'doctor',
       'laborer':'labourer','storeman':'warehouse','storekeeper':'warehouse'}
IRREG = {w: w for w in ['staff','boss','gas','glass','sales','business','process','analysis','series',
                        'works','goods','news','premises']}

def base_clean(s):
    s = str(s or '').replace('–','-').replace('—','-').replace('’',"'")
    s = GENDER.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def _sing(w):
    if w in IRREG: return w
    if len(w) > 4 and w.endswith('ies'): return w[:-3] + 'y'
    if len(w) > 4 and w.endswith('ses'): return w[:-2]
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss'): return w[:-1]
    return w

def norm(s):
    out = []
    for w in re.sub(r'[^a-z0-9\s]', ' ', str(s or '').lower()).split():
        if w in STOP: continue
        w = SYN.get(_sing(w), _sing(w))
        out += [p for p in w.split() if p not in STOP]
    return ' '.join(out)

def trailing_paren(s):
    s = base_clean(s)
    m = re.search(r'\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*$', s)
    if not m: return None
    inner = m.group(1).strip()
    if len(inner) <= 2 or NOTE.search(inner) or not re.search(r'[A-Za-z]{3}', inner): return None
    return inner

def vacancy_key(title):
    p = trailing_paren(title)
    s = p if p else base_clean(title)
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = re.sub(r'\b\d+\s*[-–]?\s*\d*\s*(?:%|hours?|hrs?|h/week|h)\b', ' ', s, flags=re.I)
    s = NOISE.sub(' ', s)
    s = re.split(r'\s+[-–—]\s+', s)[0]
    parts = [x.strip() for x in s.split(',')]
    if len(parts) > 1 and len(parts[0].split()) >= 2: s = parts[0]
    s = re.sub(r"[^A-Za-z0-9\s/+&.'-]", ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

LOC = re.compile(
    r'^(?:ksa|k\.s\.a\.?|saudi(?:\s*arabia(?:n)?)?|riyadh|jeddah|dammam|al\s*khobar|khobar|dhahran|jubail|'
    r'yanbu|neom|tabuk|abha|hail|qassim|makkah|mecca|madinah|medina|taif|najran|jazan|eastern\s*province|'
    r'western\s*region|central\s*region|red\s*sea|remote|onsite|on\s*site|hybrid|urgent|urgently|'
    r'immediate\s*joiner?s?|multiple\s*(?:positions|vacancies)|various\s*locations?|gulf|gcc|middle\s*east|'
    r'\d[\d\s\-+]*(?:years?|yrs?)?)(?:\s*(?:,|/|&|and|-)\s*.*)?$', re.I)
TRAIL_JUNK = re.compile(r'[\s*\-–—:,/|]+$')

def sa_key(title):
    s = base_clean(title)
    s = re.sub(r'\([^)]*\)', ' ', s); s = NOISE.sub(' ', s); s = re.sub(r'\s+', ' ', s).strip()
    parts = [TRAIL_JUNK.sub('', p.strip()) for p in re.split(r'\s*[-–—]\s+|\s+[-–—]\s*', s) if p.strip()]
    parts = [p for p in parts if p] or [s]
    s = ' '.join([parts[0]] + [p for p in parts[1:] if not LOC.match(p)])
    cs = [x.strip() for x in s.split(',')]
    if len(cs) > 1: s = ' '.join([cs[0]] + [c for c in cs[1:] if not LOC.match(c)])
    s = TRAIL_JUNK.sub('', s)
    s = re.sub(r"[^A-Za-z0-9\s/+&.'-]", ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def sa_norm(t): return norm(sa_key(t)) or norm(base_clean(t))

PRE = re.compile(r'\b(?:h\s*/\s*f|f\s*/\s*h|m\s*/\s*f|f\s*/\s*m|m\s*/\s*w|w\s*/\s*m|h/f/x)\b|\(e\)|\(s\)', re.I)
LOCS = re.compile(r'\b(?:dakar|abidjan|casablanca|rabat|tanger|marrakech|cairo|nairobi|mombasa|kisumu|nakuru|'
    r'eldoret|lagos|accra|johannesburg|cape town|durban|pretoria|tunis|alger|algiers|douala|yaounde|kinshasa|'
    r'kigali|kampala|dar es salaam|addis ababa|ouagadougou|bamako|lome|cotonou|niamey|conakry|libreville|'
    r'antananarivo|maputo|lusaka|harare|gaborone|windhoek|luanda|abuja|port harcourt|sandton|centurion|midrand|'
    r'vienna|wien|graz|linz|salzburg|innsbruck|egypt|morocco|maroc|senegal|kenya|nigeria|ghana|south africa|'
    r'tunisia|tunisie|algeria|cameroon|cameroun|cote d.?ivoire|ivory coast|uganda|tanzania|rwanda|ethiopia|'
    r'zambia|zimbabwe|botswana|namibia|mozambique|angola|madagascar|mauritius|burkina faso|mali|togo|benin|'
    r'niger|guinea|gabon|drc|austria|remote|hybrid)\b', re.I)

def af_norm(t):
    t = ''.join(c for c in unicodedata.normalize('NFKD', str(t or '')) if not unicodedata.combining(c))
    return sa_norm(LOCS.sub(' ', PRE.sub(' ', t)))

def af_norm2(t):
    s = str(t or '')
    if len(s.split(' – ')) >= 2: s = s.split(' – ')[0]   # "Title – Place – Department" style
    return af_norm(s.replace('&amp;', '&'))

def title_keys(t):
    """All the normalised forms a title may have been stored under, most specific first."""
    ks = [sa_norm(t), af_norm(t), af_norm2(t), norm(vacancy_key(t)), norm(base_clean(t))]
    if trailing_paren(t):          # "Job (m/f/d) (Standard occupation)" style: the bracket names the occupation
        ks.insert(0, norm(vacancy_key(t)))
    return [k for k in dict.fromkeys(ks) if k]

# ---------- similarity index over reviewed titles + official KeSCO titles ----------
def deinvert(s):
    parts = [p.strip() for p in re.split(r'[,:]', str(s or '')) if p.strip()]
    return s if len(parts) <= 1 else ' '.join(parts[1:][::-1] + [parts[0]])

_pool_keys  = [k for k in TITLE_DEC if k]
_pool_codes = [TITLE_DEC[k] for k in _pool_keys]
for c, t in KESCO_TITLES.items():           # official titles too, e.g. "Teacher, Kindergarten" -> "kindergarten teacher"
    k = norm(deinvert(t))
    if k: _pool_keys.append(k); _pool_codes.append(c)
_wv = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True)
_cv = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)
_Pw = normalize(_wv.fit_transform(_pool_keys)); _Pc = normalize(_cv.fit_transform(_pool_keys))

def nearest_titles(keys):
    Tw = normalize(_wv.transform(keys)); Tc = normalize(_cv.transform(keys))
    out = []
    for s in range(0, len(keys), 40):
        sim = (0.6 * (Tw[s:s+40] @ _Pw.T) + 0.4 * (Tc[s:s+40] @ _Pc.T)).toarray().astype('float32')
        for row in sim:
            i = int(row.argmax()); out.append((_pool_keys[i], _pool_codes[i], float(row[i])))
    return out

def code_titles(titles):
    """Returns {title: (code6, status, matched_on)}"""
    res, todo = {}, []
    for t in titles:
        if str(t).strip() in RAW_DEC:          # titles in non-Latin scripts (e.g. Amharic) stored as written
            res[t] = (RAW_DEC[str(t).strip()], 'exact', str(t).strip()); continue
        hit = next((k for k in title_keys(t) if k in TITLE_DEC), None)
        if hit: res[t] = (TITLE_DEC[hit], 'exact', hit)
        elif title_keys(t): todo.append(t)
        else: res[t] = (None, 'REVIEW (empty title)', '')
    if todo:
        keys = [title_keys(t)[0] for t in todo]
        for t, k, (src, code, sim) in zip(todo, keys, nearest_titles(keys)):
            a, b = set(src.split()), set(k.split())
            contained = bool(a and b and (a <= b or b <= a) and (len(a) >= 2 or sim >= 0.95))
            ok = (sim >= 0.85 and contained) or sim >= 0.92
            res[t] = (code, 'close match' if ok else f'REVIEW (similarity {sim:.2f})', src)
    return res

# ---------- company matching ----------
def clean_company(c):
    return re.sub(r'\s+', ' ', str(c or '').replace('&amp;', '&')).strip()

_COMP = {}
for name, code in COMPANY_DEC.items():
    _COMP.setdefault(clean_company(name).lower(), code)
def _cnorm(c):
    c = clean_company(c).lower()
    c = re.sub(r'\b(plc|p\.?\s?l\.?\s?c|ltd|limited|llc|l\.?l\.?c|inc|sarl|suarl|sa|s\.a|sas|pty|co|company|group|'
               r'corporation|corp|holding|holdings|international|s\.c|one member|share company)\b\.?', ' ', c)
    return re.sub(r'[^a-z0-9 ]', ' ', re.sub(r'\s+', ' ', c)).strip()
_cn_keys = []; _cn_codes = []
for name, code in _COMP.items():
    k = _cnorm(name)
    if k: _cn_keys.append(k); _cn_codes.append(code)
_ccv = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)
_Cm = normalize(_ccv.fit_transform(_cn_keys)) if _cn_keys else None

# Keyword fallback for companies never seen before (checked in order; first hit wins).
KEYWORDS = [
    (r'government|ministry|department of|municipal|county|city of|provincial|national treasury|public service', '841121'),
    (r'embassy|consulate', '990020'),
    (r'\bun\b|united nations|unicef|undp|unesco|world bank|african union|\bwho\b', '990010'),
    (r'recruit|staffing|manpower|talent|\bhr\b|human resource|placement|headhunt|jobs?\b|career', '781000'),
    (r'interim|temp(orary)? staff|outsourc', '782000'),
    (r'call ?cent|contact cent|\bbpo\b|telemarket', '822000'),
    (r'hotel|resort|lodge|suites|\binn\b|guest ?house', '551011'),
    (r'restaurant|burger|pizza|grill|cafe|coffee shop|bistro|kitchen', '561011'),
    (r'catering', '562100'),
    (r'\bbank\b|banque', '641200'),
    (r'microfinance|sacco|savings|credit|lending|loan', '649210'),
    (r'insurance|assurance|assur\b', '651290'),
    (r'pay(ments?)?\b|fintech|mobile money|remit', '661920'),
    (r'university|college|polytechnic|institute of technology', '853010'),
    (r'school|academy|lycee|lyc[eé]e|kindergarten|nursery', '852100'),
    (r'training|tutor|learning|education', '855000'),
    (r'hospital', '861000'),
    (r'clinic|medical cent|medical centre|health ?care|dental|diagnostic', '862010'),
    (r'pharmacy|pharmacie|chemist', '477210'),
    (r'pharma|laborator(y|ies)|biotech', '210020'),
    (r'foundation|ngo|charity|relief|humanitarian|aid\b|association|initiative', '889900'),
    (r'construction|contractor|builders?|btp|civil works', '410020'),
    (r'engineering|consulting engineers', '711020'),
    (r'real estate|realty|properties|property|developments?|immo', '682010'),
    (r'logistics|freight|shipping|cargo|transit|forwarding', '522920'),
    (r'courier|delivery|express', '532000'),
    (r'transport|haulage|trucking', '492310'),
    (r'telecom|mobile|wireless|internet|network', '612090'),
    (r'software|tech|digital|\bit\b|systems|solutions|data|cloud|cyber', '620210'),
    (r'advertis|marketing|media|brand|creative|communication', '731010'),
    (r'consult', '702000'),
    (r'security|guard', '801000'),
    (r'cleaning|facility|facilities', '812100'),
    (r'oil|gas|petroleum|energy services', '091012'),
    (r'solar|energy|power', '351013'),
    (r'mining|mines|minerals|gold', '072900'),
    (r'motors?\b|auto(motive)?|cars?\b|vehicle', '451010'),
    (r'farm|agri|agro|poultry', '015000'),
    (r'food|foods|beverage|drinks|dairy|bakery|bread', '107900'),
    (r'furniture', '310010'),
    (r'textile|garment|apparel|fashion|clothing', '141000'),
    (r'plastic', '222010'),
    (r'print', '181190'),
    (r'travel|tours?\b', '791100'),
    (r'supermarket|store|stores|shop|retail|mart\b|market\b', '471190'),
    (r'trading|import|export|general trading|distribut', '461000'),
    (r'law firm|lawyers?|advocates|legal|avocat', '691000'),
    (r'account(ing|ants)|audit', '692010'),
]
KEYWORDS = [(re.compile(p, re.I), c) for p, c in KEYWORDS]

def code_company(c):
    """Returns (code6 or None, status, matched_on)"""
    raw = clean_company(c)
    if not raw or raw.lower() in UNNAMED:
        return None, 'blank (unnamed)', ''
    for name in (str(c).strip(), raw):
        if name in COMPANY_DEC:
            return COMPANY_DEC[name], 'exact', name
    if raw.lower() in _COMP:
        return _COMP[raw.lower()], 'exact', raw
    k = _cnorm(raw)
    if k and _Cm is not None:
        row = (normalize(_ccv.transform([k])) @ _Cm.T).toarray()[0]
        i = int(row.argmax())
        if row[i] >= 0.90:
            return _cn_codes[i], 'close match', _cn_keys[i]
    for rx, code in KEYWORDS:
        if rx.search(raw):
            return code, 'REVIEW (keyword)', rx.pattern
    return None, 'REVIEW (unknown company - left blank)', ''

print("Matching rules ready.")