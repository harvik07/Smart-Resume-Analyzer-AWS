# Resume Score

Resume Score is a web application that analyzes and scores resumes against job descriptions using ATS (Applicant Tracking System) criteria. It provides actionable suggestions to improve your resume's match with a given job description.

## Features
- Upload your resume (PDF) and job description to receive an ATS compatibility score
- Highlights missing sections and keywords
- Provides suggestions to improve your resume
- Editor mode for direct resume text input and scoring
- Supports synonym and section detection using NLP
- Built with Flask, spaCy, and Sentence Transformers

## How It Works
1. Upload your resume and paste the job description.
2. The app extracts text from your resume and analyzes it using NLP.
3. It compares your resume to the job description for section coverage, keyword match, and ATS-friendliness.
4. You receive a score and suggestions for improvement.



## Dependencies
- Flask
- sentence-transformers
- pymupdf
- scikit-learn
- nltk
- language-tool-python
- spacy

