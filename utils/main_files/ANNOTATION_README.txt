Legal Case Annotation Tool
=========================

This tool allows legal professionals to annotate case documents with structured information including:
- Basic case details (court, judgment date, etc.)
- Cited cases with detailed references
- Final decisions
- Case types
- Statutes and legal references
- Judges and jurisdictions

Getting Started
--------------

1. Run the annotation tool:
   ```
   python run_annotation_tool.py
   ```
   
2. The Streamlit interface will open in your web browser.

3. Choose to either:
   - Load an existing case JSON file
   - Create a new case

4. Use the various tabs to annotate different aspects of the case:
   - Basic case information
   - Case content
   - Cited cases (with citation and relevance)
   - Statutes referenced
   - Judges involved
   - Jurisdictions
   - Additional metadata

5. After making changes, click "Save Annotations" to save your work.

Schema
------
The case data follows a structured JSON schema (see utils/main_files/case_schema.json) with the following main sections:

- id: Unique identifier for the case
- title: Title of the case
- court: Court that heard the case
- judgment_date: Date of the judgment
- content: Full text content of the case
- entities: Extracted entities (cases, statutes, judges, jurisdictions)
- metadata: Additional metadata
- cited_cases: Detailed information about cited cases
- final_decision: Summary of the final decision/ruling
- case_type: Type or category of the case

Usage Notes
-----------
1. When adding cited cases, you can specify both the case name and its citation.
2. The "final_decision" field should contain a concise summary of the court's ruling.
3. The "case_type" field should specify the category of the case (e.g., Civil Appeal, Criminal).

Data is saved in JSON format, making it easy to integrate with other systems and analysis tools. 