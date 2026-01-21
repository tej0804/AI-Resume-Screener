# AI Resume Screening & Candidate Ranking System

An end-to-end AI-powered resume screening system that evaluates, ranks, and shortlists candidates against job descriptions using rule-based signals + NLP-based semantic matching, with automatic interview question generation.

This project simulates how modern ATS (Applicant Tracking Systems) and AI hiring platforms operate in real-world recruitment pipelines.

## Project Goal

To build an interpretable and scalable AI system that:

1. Screens resumes automatically
2. Matches candidates to job descriptions
3. Ranks candidates using hybrid AI scoring
4. Evaluates ranking quality
5. Generates interview questions based on skills

## End-to-End Pipeline
Resumes + Job Descriptions
        ->
Text Cleaning & Normalization
        ->
Skill / Experience / Education Extraction
        ->
Feature Engineering
    Rule-based scoring,
    TF-IDF similarity,
    SBERT semantic similarity
        ->
Hybrid Weighted Scoring
        ->
Candidate Ranking
        ->
Evaluation (Precision@K)
        ->
Interview Question Generation



##  Phase 1: Data Preprocessing

Implemented:

1. Text normalization (lowercase, punctuation removal, cleanup)
2. Resume and JD parsing
3. Structured information extraction
4. Extracted Features
5. Skills (keyword + NLP-based)
6. Education level (Undergraduate / Postgraduate / PhD)
7. Experience (years → junior / mid / senior)

Output:

processed_resumes.pkl, processed_jds.pkl

## Phase 2: Feature Engineering

### a. Rule-Based Matching, Each resume is scored against a job description using:

1. Skill overlap score
2. Experience match score
3. Education alignment score
4. Weighted rule score:

rule_score = 0.6 × skill_match + 0.3 × experience_match + 0.1 × education_match


This ensures human-readable and explainable decisions, critical for recruitment systems.

### b. Text-Based Similarity Features -> TF-IDF (Baseline)

1. Captures keyword overlap
2. Provides lexical matching benchmark

### c. SBERT (Semantic Matching)

1. Captures contextual and semantic similarity
2. Handles paraphrasing and meaning-based matching

## Phase 3: Hybrid Candidate Ranking

Final candidate score is computed using a hybrid weighted model:

final_score = 0.30 × rule_score + 0.20 × TF-IDF_similarity + 0.50 × SBERT_similarity


This balances:
Interpretability (rules), Keyword relevance (TF-IDF), Deep semantic understanding (SBERT)

## Phase 4: Evaluation (Phase 5 – Critical)

Metrics Implemented:

1. Precision@K (K = 10)
2. Comparisons Performed
3. TF-IDF only ranking
4. SBERT only ranking
5. Hybrid ranking

Synthetic labels are used and clearly documented, which is acceptable for portfolio and academic projects.

## Phase 5: Interview Question Auto-Generation

Automatically generates interview questions based on extracted skills

Focuses on:

1. Core skills
2. Missing or weak skill areas compared to the JD

Example
Skill: Machine Learning     Question: Explain the bias–variance tradeoff with a real-world example.
This simulates AI-assisted recruiter decision support.

Output:

Final ranked candidates are saved with:
1. Rule-based score
2. TF-IDF similarity
3. SBERT similarity
4. Final hybrid score
5. Rank position


## Skills Demonstrated

1. Natural Language Processing (TF-IDF, SBERT)
2. Feature engineering for ranking systems
3. Hybrid ML + rule-based modeling
4. Ranking evaluation metrics (Precision@K)
5. Explainable AI for hiring systems
6. End-to-end applied machine learning

### Why This Project Matters?

This project demonstrates:

1. Real-world ATS-style resume screening
2. Explainable + semantic AI combination
3. Ranking evaluation (not just classification)
4. Practical AI use in recruitment workflows
