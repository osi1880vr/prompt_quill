import random
import re
from .wildcards import WildcardCache
import shared

class PromptEnhance:
    def __init__(self):
        self.cache = WildcardCache()
        self.face_features = [
            "with a sharp jawline", "sporting curly hair", "with piercing blue eyes",
            "having a crooked smile", "with a freckled face", "sporting a nose piercing"
            "with high cheekbones", "having a scar over one eye", "with a wide grin",
            "sporting a pointed goatee", "with deep-set green eyes", "having a dimpled chin",
            "with deep-set blue eyes", "with piercing grey eyes", "with deep-set brown eyes",
            "with a surprised look"
        ]

    def get_max_from_filename(self, filename):
        match = re.match(r"(.+)_(\d+)\.txt$", filename)
        if match:
            return int(match.group(2))
        return None

    def get_singular(self, word):
        """Convert a plural noun to its singular form with high accuracy."""

        # Comprehensive list of irregular plurals
        irregulars = {
            "men": "man", "women": "woman", "children": "child", "teeth": "tooth",
            "feet": "foot", "mice": "mouse", "geese": "goose", "oxen": "ox",
            "cacti": "cactus", "fungi": "fungus", "nuclei": "nucleus",
            "alumni": "alumnus", "radii": "radius", "bacteria": "bacterium",
            "phenomena": "phenomenon", "criteria": "criterion", "data": "datum",
            "media": "medium", "stimuli": "stimulus", "formulae": "formula",
            "vertebrae": "vertebra", "larvae": "larva"
        }

        # Clean the input word: Remove non-alphabetic characters (like punctuation)
        word = re.sub(r"[^a-zA-Z]", "", word).strip()

        # Direct match for known irregular words
        if word in irregulars:
            return irregulars[word]

        # Handle common pluralization rules using regex
        if re.search(r"ies$", word) and len(word) > 3:  # e.g., "puppies" -> "puppy"
            return re.sub(r"ies$", "y", word)

        if re.search(r"ves$", word):  # e.g., "wolves" -> "wolf"
            return re.sub(r"ves$", "f", word)

        if re.search(r"oes$", word) and word not in {"shoes", "heroes"}:  # e.g., "potatoes" -> "potato"
            return re.sub(r"oes$", "o", word)

        if re.search(r"xes$", word) and word not in {"taxes", "axes"}:  # e.g., "boxes" -> "box"
            return re.sub(r"es$", "", word)

        if re.search(r"sses$", word):  # Avoid changing words like "glasses"
            return word

        if re.search(r"es$", word) and not re.search(r"[sxz]es$", word):  # e.g., "foxes" -> "fox" but not "glasses"
            return re.sub(r"es$", "", word)

        if re.search(r"s$", word) and not re.search(r"[us]s$", word):  # e.g., "dogs" -> "dog", but not "mass" or "bonus"
            return re.sub(r"s$", "", word)

        return word  # No change if no rule applies



    def clean_prompt(self, prompt, options):
        """Remove existing enhancements from the prompt, considering multi-word phrases."""
        # Sort options by length (longest first) to avoid partial match issues
        sorted_options = sorted(options, key=len, reverse=True)

        for option in sorted_options:
            # Remove exact multi-word matches (preserving spaces and commas properly)
            prompt = re.sub(rf",?\s*\b{re.escape(option)}\b\s*,?", ",", prompt)

        # Clean up any leftover double commas or trailing commas
        prompt = re.sub(r",\s*,", ",", prompt)  # Fix double commas
        prompt = prompt.strip().strip(",")  # Remove trailing commas

        return prompt

    def enhance_prompt(self, prompt):
        files = self.cache.get_file_list()
        keywords = {f[:-4].split("_")[0]: f for f in files}
        keyword_files = {f[:-4].split("_")[0]: f for f in files}

        words = prompt.lower().split()
        used_enhancements = set(part.strip() for part in prompt.split(",") if part.strip())  # Track existing enhancements

        for word in words:
            filename = None
            if word in keywords:
                filename = keyword_files[word]
            else:
                singular = self.get_singular(word)
                if singular in keywords and word != singular:
                    filename = keyword_files[singular]

            if filename:
                options = self.cache.load_wildcards(filename)
                if options:
                    # Clean out old enhancements from this file
                    cleaned_prompt = self.clean_prompt(prompt, options)
                    # Filter available options, excluding any still in prompt or used
                    available_options = [opt for opt in options if opt not in used_enhancements]
                    if available_options:
                        max_limit = self.get_max_from_filename(filename)
                        num_additions = random.randint(1, len(available_options))
                        if max_limit is not None and num_additions > max_limit:
                            num_additions = max_limit
                        chosen = random.sample(available_options, num_additions)
                        used_enhancements.update(chosen)
                        additions = ", ".join(chosen)
                        prompt = f"{cleaned_prompt}, {additions}"

        return prompt

    def detect_people_count(self, prompt):
        prompt_lower = prompt.lower()

        # Number words mapping
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
        }

        # Extract text between commas
        comma_phrases = re.findall(r",\s*([^,]+)\s*,", prompt_lower)

        # List of people-related words
        people_terms = ["people", "men", "women", "friends", "children", "girls", "boys", "guys", "ladies", "gentlemen"]

        # Search for a number + people term (with or without an extra word in between)
        for phrase in comma_phrases:
            words = phrase.split()
            for i in range(len(words) - 1):
                # Check if the current word is a number word
                if words[i] in number_words:
                    # Next word or the one after should be a people term
                    if i + 1 < len(words) and words[i + 1] in people_terms:
                        return number_words[words[i]]
                    if i + 2 < len(words) and words[i + 2] in people_terms:
                        return number_words[words[i]]

        # Check for plural words suggesting multiple people
        if any(term in prompt_lower for term in ["people", "friends", "crowd", "group"]):
            return 2  # Assume at least two if a general plural is used

        # Detect conjunction "and" (e.g., "a man and a woman")
        if " and " in prompt_lower:
            people_terms = ["man", "woman", "person", "friend", "child"]
            matches = [term for term in people_terms if term in prompt_lower]
            if len(matches) >= 2:
                return 2

        # If only a single person is mentioned
        if any(term in prompt_lower for term in ["man", "woman", "person", "friend", "child"]):
            return 1

        return 0  # Default if no people are detected

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

