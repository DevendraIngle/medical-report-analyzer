import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import black, red, maroon 
import re
class PDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        

        
        self.styles.add(ParagraphStyle(name='ReportTitle',
                                        fontSize=18,
                                        leading=22,
                                        alignment=TA_CENTER,
                                        spaceAfter=18,
                                        textColor=black,
                                        bold=True))

        self.styles.add(ParagraphStyle(name='HeaderInfo',
                                        fontSize=10,
                                        leading=14,
                                        alignment=TA_LEFT,
                                        spaceAfter=3,
                                        textColor=black))

        self.styles.add(ParagraphStyle(name='SectionTitle',
                                        fontSize=14,
                                        leading=18,
                                        spaceBefore=16,
                                        spaceAfter=8,
                                        bold=True,
                                        textColor=black))

        
        self.styles.add(ParagraphStyle(name='AnalysisBodyText', # Renamed from BodyText
                                        parent=self.styles['Normal'], # Inherit from Normal style
                                        fontSize=10,
                                        leading=14,
                                        spaceAfter=6, # Space between paragraphs
                                        textColor=black))

        
        self.styles.add(ParagraphStyle(name='Disclaimer',
                                        fontSize=9,
                                        leading=11,
                                        spaceBefore=24,
                                        textColor=maroon,
                                        alignment=TA_LEFT,
                                        parent=self.styles['Normal']))


    def generate_pdf(self, analysis_text, user_profile, report_metadata):
        """Generates a PDF from the analysis text, including user and report metadata."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=inch/2, leftMargin=inch/2,
                                topMargin=inch/2, bottomMargin=inch/2)

        story = []

        story.append(Paragraph("Medical Report Analysis (AI Generated)", self.styles['ReportTitle']))

        story.append(Paragraph(f"<b>User:</b> {user_profile.get('name', 'N/A')}", self.styles['HeaderInfo']))
        story.append(Paragraph(f"<b>Report File:</b> {report_metadata.get('filename', 'N/A')}", self.styles['HeaderInfo']))
        story.append(Paragraph(f"<b>Report Type:</b> {report_metadata.get('report_type', 'Unknown')}", self.styles['HeaderInfo']))
        story.append(Paragraph(f"<b>Upload Date:</b> {report_metadata.get('upload_date', 'N/A')}", self.styles['HeaderInfo']))
        story.append(Spacer(1, 0.2*inch)) # Space after header info


        
        final_disclaimer_match = re.search(r'\n*Important Disclaimer:\s*(.*)$', analysis_text, re.DOTALL) # Added \n* to handle optional leading newlines

        main_analysis_body = analysis_text
        final_disclaimer_text = None

        if final_disclaimer_match:
             main_analysis_body = analysis_text[:final_disclaimer_match.start()].strip()
             final_disclaimer_text = final_disclaimer_match.group(1).strip()


        
        section_pattern = r'\n\*\*(\d+\.\s[^:]+:?)\*\*\n'

        sections_split = re.split(section_pattern, main_analysis_body)

        
        if sections_split and sections_split[0].strip():
             processed_text = self._format_text_for_paragraph(sections_split[0].strip())
             story.append(Paragraph(processed_text, self.styles['AnalysisBodyText']))
             story.append(Spacer(1, 0.1*inch))


        
        for i in range(1, len(sections_split), 2):
            if i + 1 < len(sections_split):
                 heading_part = sections_split[i].strip() 
                 content = sections_split[i+1].strip()

                 
                 story.append(Paragraph(heading_part, self.styles['SectionTitle']))

                 
                 paragraphs_content = content.split('\n\n')

                 for para_content in paragraphs_content:
                      if para_content.strip():
                          
                          processed_para_text = self._format_text_for_paragraph(para_content.strip())
                          # CORRECTED: Use 'AnalysisBodyText' style
                          story.append(Paragraph(processed_para_text, self.styles['AnalysisBodyText']))
                      # Spacer is handled by the spaceAfter in BodyText style

                 
                 if i + 2 < len(sections_split):
                     story.append(Spacer(1, 0.2*inch))


        if final_disclaimer_text:
             
             processed_disclaimer_text = self._format_text_for_paragraph(final_disclaimer_text)
             story.append(Paragraph(f"<b>Important Disclaimer:</b> {processed_disclaimer_text}", self.styles['Disclaimer']))
        else:
            
             story.append(Paragraph("<b>Important Disclaimer:</b> This analysis is AI-generated and for informational purposes only. Always consult a medical professional.", self.styles['Disclaimer']))


        try:
            doc.build(story)
            buffer.seek(0)
            return buffer
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            
            return None

    def _format_text_for_paragraph(self, text):
        """Applies formatting rules for ReportLab Paragraphs: **text** to <b>text</b>, single \n to <br/>."""
        
        formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

        
        formatted_text = formatted_text.replace('\n', '<br/>')


        return formatted_text
