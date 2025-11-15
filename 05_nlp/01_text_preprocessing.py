"""
NLP Basics - Text Preprocessing
Complete text preprocessing pipeline for NLP tasks.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    """Complete text preprocessing pipeline"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def to_lowercase(self, text):
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_emails(self, text):
        """Remove email addresses"""
        email_pattern = r'\S+@\S+'
        return re.sub(email_pattern, '', text)
    
    def remove_special_chars(self, text, keep_punctuation=False):
        """Remove special characters"""
        if keep_punctuation:
            # Keep only alphanumeric and basic punctuation
            return re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
        else:
            # Keep only alphanumeric and spaces
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def remove_numbers(self, text):
        """Remove numbers"""
        return re.sub(r'\d+', '', text)
    
    def remove_extra_spaces(self, text):
        """Remove extra whitespace"""
        return ' '.join(text.split())
    
    def tokenize_words(self, text):
        """Tokenize text into words"""
        try:
            return word_tokenize(text)
        except:
            nltk.download('punkt')
            return word_tokenize(text)
    
    def tokenize_sentences(self, text):
        """Tokenize text into sentences"""
        try:
            return sent_tokenize(text)
        except:
            nltk.download('punkt')
            return sent_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stop words"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem(self, tokens):
        """Apply stemming"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize(self, tokens):
        """Apply lemmatization"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text, 
                   lowercase=True,
                   remove_urls=True,
                   remove_emails=True,
                   remove_special_chars=True,
                   remove_numbers=False,
                   remove_stopwords=True,
                   lemmatize=True,
                   stem=False):
        """
        Complete preprocessing pipeline
        """
        # Step 1: Lowercase
        if lowercase:
            text = self.to_lowercase(text)
        
        # Step 2: Remove URLs
        if remove_urls:
            text = self.remove_urls(text)
        
        # Step 3: Remove emails
        if remove_emails:
            text = self.remove_emails(text)
        
        # Step 4: Remove special characters
        if remove_special_chars:
            text = self.remove_special_chars(text, keep_punctuation=False)
        
        # Step 5: Remove numbers (optional)
        if remove_numbers:
            text = self.remove_numbers(text)
        
        # Step 6: Remove extra spaces
        text = self.remove_extra_spaces(text)
        
        # Step 7: Tokenize
        tokens = self.tokenize_words(text)
        
        # Step 8: Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Step 9: Lemmatize or Stem
        if lemmatize:
            tokens = self.lemmatize(tokens)
        elif stem:
            tokens = self.stem(tokens)
        
        return tokens


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Text Preprocessing Pipeline")
    print("=" * 60)
    
    # Sample text
    sample_text = """
    Hello! This is a sample text for NLP preprocessing.
    Visit us at https://example.com or email john@example.com
    We have 100 products and amazing features!!!
    The cats are running and playing. They ran quickly.
    """
    
    print("\nOriginal Text:")
    print(sample_text)
    
    # Create preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess
    processed = preprocessor.preprocess(
        sample_text,
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_special_chars=True,
        remove_numbers=False,
        remove_stopwords=True,
        lemmatize=True
    )
    
    print("\nPreprocessed Tokens:")
    print(processed)
    
    print("\nPreprocessed Text (joined):")
    print(' '.join(processed))
    
    # Compare with stemming
    print("\n" + "-" * 60)
    print("Comparison: Lemmatization vs Stemming")
    print("-" * 60)
    
    tokens = preprocessor.tokenize_words(sample_text.lower())
    tokens = preprocessor.remove_stopwords(tokens)
    
    lemmatized = preprocessor.lemmatize(tokens)
    stemmed = preprocessor.stem(tokens)
    
    print("\nOriginal -> Lemmatized -> Stemmed")
    for orig, lem, stem in zip(tokens[:10], lemmatized[:10], stemmed[:10]):
        print(f"{orig:15} -> {lem:15} -> {stem}")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Apply to real datasets")
    print("2. Experiment with different preprocessing options")
    print("3. Move to text representation (BoW, TF-IDF, Word2Vec)")
    print("=" * 60)

