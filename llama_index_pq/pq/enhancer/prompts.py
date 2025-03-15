import random
import re
from .wildcards import WildcardCache
import shared

class PromptEnhance:
    def __init__(self):
        self.cache = WildcardCache()
        self.face_features = [
            "with a sharp jawline", "sporting curly hair", "with piercing blue eyes",
            "having a crooked smile", "with a freckled face", "sporting a nose piercing",
            "with high cheekbones", "having a scar over one eye", "with a wide grin",
            "sporting a pointed goatee", "with deep-set green eyes", "having a dimpled chin",
            "with deep-set blue eyes", "with piercing grey eyes", "with deep-set brown eyes",
            "with a surprised look"
        ]
        # Ensure swap categories are loaded on init
        self.cache.load_auto_swap_categories()  # Load dynamic categories on init

    def get_max_from_filename(self, filename):
        match = re.match(r"(.+)_(\d+)\.txt$", filename)
        if match:
            return int(match.group(2))
        return None

    def get_singular(self, word):
        """Convert a plural noun to its singular form with high accuracy."""
        irregulars = {
            "men": "man", "women": "woman", "children": "child", "teeth": "tooth",
            "feet": "foot", "mice": "mouse", "geese": "goose", "oxen": "ox",
            "cacti": "cactus", "fungi": "fungus", "nuclei": "nucleus",
            "alumni": "alumnus", "radii": "radius", "bacteria": "bacterium",
            "phenomena": "phenomenon", "criteria": "criterion", "data": "datum",
            "media": "medium", "stimuli": "stimulus", "formulae": "formula",
            "vertebrae": "vertebra", "larvae": "larva"
        }
        word = re.sub(r"[^a-zA-Z]", "", word).strip()
        if word in irregulars:
            return irregulars[word]
        if re.search(r"ies$", word) and len(word) > 3:
            return re.sub(r"ies$", "y", word)
        if re.search(r"ves$", word):
            return re.sub(r"ves$", "f", word)
        if re.search(r"oes$", word) and word not in {"shoes", "heroes"}:
            return re.sub(r"oes$", "o", word)
        if re.search(r"xes$", word) and word not in {"taxes", "axes"}:
            return re.sub(r"es$", "", word)
        if re.search(r"sses$", word):
            return word
        if re.search(r"es$", word) and not re.search(r"[sxz]es$", word):
            return re.sub(r"es$", "", word)
        if re.search(r"s$", word) and not re.search(r"[us]s$", word):
            return re.sub(r"s$", "", word)
        return word

    def clean_prompt(self, prompt, options):
        """Remove existing enhancements from the prompt, considering multi-word phrases."""
        sorted_options = sorted(options, key=len, reverse=True)
        for option in sorted_options:
            prompt = re.sub(rf",?\s*\b{re.escape(option)}\b\s*,?", ",", prompt)
        prompt = re.sub(r",\s*,", ",", prompt)
        prompt = prompt.strip().strip(",")
        return prompt

    def swap_category_terms(self, prompt, category):
        """Swap exact terms or phrases from a specific category with consistent random alternatives."""
        options = self.cache.get_auto_swap_options(category)
        if not options:
            return prompt

        # Build a replacement dictionary for this category
        pattern = r"(?<!\w)(" + "|".join(re.escape(opt) for opt in options) + r")(?!\w)"
        matches = list(re.finditer(pattern, prompt.lower()))

        if not matches:
            return prompt

        # Group by term and pick one replacement per term
        replacements = {}
        for match in matches:
            original_term = match.group(1)
            if original_term not in replacements:
                available_options = [opt for opt in options if opt.lower() != original_term.lower()]
                replacements[original_term] = random.choice(available_options) if available_options else original_term

        # Debug replacements
        print(f"Category: {category}")
        for orig, new in replacements.items():
            print(f"Replacing '{orig}' with '{new}'")

        # Apply all replacements in one pass using a case-preserving approach
        def replace_match(match):
            orig = match.group(0)  # Full match in original case
            lower_orig = orig.lower()
            new_term = replacements.get(lower_orig, orig)
            # Preserve original case if possible
            if orig.isupper():
                return new_term.upper()
            elif orig[0].isupper():
                return new_term.capitalize()
            return new_term

        # Use re.sub with the pattern and replacement function
        updated_prompt = re.sub(pattern, replace_match, prompt, flags=re.IGNORECASE)
        return updated_prompt

    def enhance_prompt(self, prompt):
        self.cache.load_auto_swap_categories()
        for category in self.cache.auto_swap_data.keys():
            prompt = self.swap_category_terms(prompt, category)

        files = self.cache.get_file_list()
        keywords = {f[:-4].split("_")[0]: f for f in files}
        keyword_files = {f[:-4].split("_")[0]: f for f in files}

        words = prompt.lower().split()
        used_enhancements = set(part.strip() for part in prompt.split(",") if part.strip())

        keyword_positions = {}
        for i, word in enumerate(words):
            filename = None
            if word in keywords:
                filename = keyword_files[word]
            else:
                singular = self.get_singular(word)
                if singular in keywords and word != singular:
                    filename = keyword_files[singular]

            if filename and filename not in [f"{cat}.txt" for cat in self.cache.auto_swap_data.keys()]:
                if filename not in keyword_positions:
                    keyword_positions[filename] = []
                keyword_positions[filename].append(i)

        for filename, positions in keyword_positions.items():
            options = self.cache.load_wildcards(filename)
            if options:
                available_options = [opt for opt in options if opt not in used_enhancements]
                if available_options:
                    max_limit = self.get_max_from_filename(filename)
                    num_additions = random.randint(1, len(available_options))
                    if max_limit is not None and num_additions > max_limit:
                        num_additions = max_limit
                    chosen = random.sample(available_options, num_additions)
                    used_enhancements.update(chosen)
                    additions = ", ".join(chosen)
                    cleaned_prompt = prompt
                    for pos in positions:
                        cleaned_prompt = self.clean_prompt(cleaned_prompt, options)
                        cleaned_prompt = f"{cleaned_prompt}, {additions}"
                    prompt = cleaned_prompt

        return prompt

    def detect_people_count(self, prompt):
        prompt_lower = prompt.lower()
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
        }
        comma_phrases = re.findall(r",\s*([^,]+)\s*,", prompt_lower)
        people_terms = ["people", "men", "women", "friends", "children", "girls", "boys", "guys", "ladies", "gentlemen"]
        for phrase in comma_phrases:
            words = phrase.split()
            for i in range(len(words) - 1):
                if words[i] in number_words:
                    if i + 1 < len(words) and words[i + 1] in people_terms:
                        return number_words[words[i]]
                    if i + 2 < len(words) and words[i + 2] in people_terms:
                        return number_words[words[i]]
        if any(term in prompt_lower for term in ["people", "friends", "crowd", "group"]):
            return 2
        if " and " in prompt_lower:
            people_terms = ["man", "woman", "person", "friend", "child"]
            matches = [term for term in people_terms if term in prompt_lower]
            if len(matches) >= 2:
                return 2
        if any(term in prompt_lower for term in ["man", "woman", "person", "friend", "child"]):
            return 1
        return 0

    def enhance_faces(self, ad_prompt, prompt, sep_prompt, used_enhancements=None):
        if used_enhancements is None:
            used_enhancements = set()
        people_count = self.detect_people_count(prompt)
        if people_count > 1:
            available_faces = [f for f in self.face_features if f not in used_enhancements]
            if len(available_faces) >= people_count:
                chosen_faces = random.sample(available_faces, people_count)
                used_enhancements.update(chosen_faces)
                face_descriptions = [f"one with {face}" for face in chosen_faces]
                ad_prompt = f"{ad_prompt}, {f'[SEP] {sep_prompt}, '.join(face_descriptions)}"
        return ad_prompt

    def process_wildcards(self, input_string):
        processor = shared.WildcardResolver()
        return processor.resolve_prompt(input_string)