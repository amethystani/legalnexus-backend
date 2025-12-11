Legal Case Annotation with Label Studio
====================================

This documentation explains how to use Label Studio for legal case annotation. Label Studio is a powerful annotation tool that allows legal professionals to annotate case documents with structured information.

Setup
-----

1. Install Label Studio:
   ```
   pip install label-studio
   ```

2. Run the Label Studio setup script:
   ```
   python run_label_studio.py
   ```

3. Access Label Studio at http://localhost:8080
   - Username: admin@example.com
   - Password: admin123

Annotation Workflow
------------------

1. **Import Cases**:
   - Import existing JSON case files into Label Studio using the converter
   - Alternatively, import raw text documents to annotate from scratch

2. **Annotate Legal Entities**:
   - Highlight and label important entities in the text:
     - Cases (cited cases)
     - Statutes (legal references)
     - Judges
     - Jurisdictions

3. **Add Case Details**:
   - Fill in the form fields for case information:
     - Basic information (title, court, date, type)
     - Final decision (summary of the ruling)
     - Cited cases with their citations and relevance
     - Additional metadata

4. **Export Annotations**:
   - Export your annotations in Label Studio format
   - Convert to our JSON schema using the converter

File Format
-----------

The annotation tool works with two file formats:

1. **Our JSON Schema**: The format used in our backend (see utils/main_files/case_schema.json)
   - Contains structured case information with nested fields
   - Includes sections for cited cases, entities, final decision, etc.

2. **Label Studio Format**: The format used for annotation in Label Studio
   - Contains the original content plus annotations
   - Supports visual labeling of entities in the text

Conversion Between Formats
-------------------------

Use the converter tool to translate between formats:

```
# Convert from our JSON to Label Studio format
python utils/main_files/label_studio_converter.py --input case.json --output case_ls.json --direction to_label_studio

# Convert from Label Studio to our JSON format
python utils/main_files/label_studio_converter.py --input case_ls.json --output case.json --direction to_json
```

Annotation Guidelines
--------------------

1. **Entity Annotation**:
   - Be precise when highlighting entities in the text
   - Make sure to capture the full name/reference

2. **Cited Cases**:
   - Add all cases referenced in the document
   - Include proper citations when available
   - Rate relevance based on how central the case is to the ruling

3. **Final Decision**:
   - Provide a concise summary of the court's ruling
   - Focus on the legal principles established

4. **Case Type**:
   - Specify the category of case (e.g., Civil Appeal, Criminal, Constitutional)

5. **Statutes**:
   - Include the full name and section of statutes referenced

Best Practices
-------------

1. Complete all required fields for each case
2. Be consistent in how you format citations
3. Check for accuracy in entity recognition
4. Save your work frequently

For more information about Label Studio, visit their documentation at: https://labelstud.io/guide/ 