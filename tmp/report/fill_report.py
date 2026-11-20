from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output' / 'pdf' / 'TransitionX_Progress_Report_Draft.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H = 595.5, 842.25
BG = HexColor('#1c1c1c')
CELL = HexColor('#2a2432')
PURPLE = HexColor('#390565')
WHITE = HexColor('#f4f1f6')
MUTED = HexColor('#d2a4df')
c = canvas.Canvas(str(OUT), pagesize=(W, H))
c.setTitle('TransitionX 2026 Project Progress Report')
style = ParagraphStyle('body', fontName='Helvetica', fontSize=9, leading=11.6, textColor=WHITE)

def clear(x, top, width, height, color=BG):
    c.setFillColor(color)
    c.rect(x, H-top-height, width, height, fill=1, stroke=0)

def text(value, x, top, width, max_height, size=9, color=WHITE):
    s = ParagraphStyle('local', parent=style, fontSize=size, leading=size*1.25, textColor=color)
    p = Paragraph(value, s)
    _, height = p.wrap(width, max_height)
    if height > max_height + .5:
        raise ValueError(f'Text overflow {height:.1f}>{max_height}: {value[:80]}')
    p.drawOn(c, x, H-top-height)
    return height

def para(label, value, top, height, x=57, width=482, size=9):
    return text('<b>'+escape(label)+'</b> '+escape(value), x, top, width, height, size)

def table(headers, rows, widths, top, heights=None, size=8):
    s = ParagraphStyle('table', parent=style, fontSize=size, leading=size*1.22)
    def cell(value):
        if value.startswith('https://'):
            label = 'Open GitHub repository' if 'github.com' in value else 'Open project plan on Google Drive'
            return Paragraph('<link href="'+escape(value, {'"':'&quot;'})+'" color="#d2a4df"><u>'+label+'</u></link>', s)
        return Paragraph(escape(value), s)
    content = [[Paragraph('<b>'+escape(v)+'</b>', s) for v in headers]]
    content += [[cell(v) for v in row] for row in rows]
    t=Table(content, colWidths=widths, rowHeights=heights)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),PURPLE),
        ('BACKGROUND',(0,1),(-1,-1),CELL),
        ('TEXTCOLOR',(0,0),(-1,-1),WHITE),
        ('GRID',(0,0),(-1,-1),.45,HexColor('#6f547d')),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    _,h=t.wrap(sum(widths),H)
    t.drawOn(c,57,H-top-h)
    return top+h

def page(n):
    c.drawImage(str(ROOT/'tmp'/'report'/f'template-{n}.png'),0,0,width=W,height=H)
    # The source PDF's media box starts 7.83 points below its artwork origin.
    c.translate(0, -7.83)

page(1)
values = [
    'CoffeeOverflow',
    'Intelligent Electronics, Robotics &amp; Embedded Systems',
    'Computer-Vision Library for Automatic Water-Leak Detection in Hydrostatic-Pressure Testing',
    'SLIIT',
    'Ardmel Manufacturing (Pvt) Ltd',
    'Ranasinghe A.B. | ranasingheanuda@gmail.com',
    'Wickramasinghe G.K.D.K.; M.D.C Yavindi<br/>All members: shared development and testing',
    'Report 01 / 05 Sep 2026 - 28 Oct 2026',
    '28 Oct 2026',
    'Mentor: Tilanka Senadheera | Academic advisor: Not assigned',
]
for i,value in enumerate(values):
    top=353.32+i*26.939
    clear(244.5,top+.5,293.8,25.9,CELL)
    text(value,251,top+(3 if i in (2,6) else 8),281,21 if i in (2,6) else 18,size=7.7 if i in (2,6) else 8.4)
c.showPage()

page(2)
clear(52,100,493,145)
para('Current problem statement / scope:',
     'Automate identification of the first three independent water-leak positions on fabric during hydrostatic-pressure testing. The hydrotester requires camera-based droplet detection, timestamps, coordinates, saved evidence and results returned to its main application. Droplets must be distinguished from texture, reflections, wet areas, noise, vibration and lighting changes (E1, p. 4).',103,61)
para('Baseline data, scale of impact and source:',
     'The industry proposal defines the need; it supplies no measured manual-test error rate, throughput or cost baseline. These operational figures require company confirmation.',170,35)
para('Changes since initial proposal:',
     'The prototype moved from dry/wet reference comparison to a trained single-image detector after missed droplets and texture-related false positives. The final live-video scope remains unchanged.',211,31)

clear(52,289,493,149)
para('Current solution approach and why it addresses the problem:',
     'Train a local YOLO detector using labelled fabric images. For each image, output droplet centres and confidence values. Planned live integration: USB camera -> detector -> temporal confirmation and position tracking -> first three independent leak events -> timestamped images, video and structured results for the hydrotester.',292,59)
para('Scope, assumptions and practical constraints:',
     'A reusable Python library must integrate with the hydrotester. Camera settings, ROI, recording paths and timing tolerance remain to be agreed and tested. Representative footage from the completed machine is essential; reliable machine-specific development and validation cannot proceed without this data.',355,46)
para('Relevant architecture / design evidence:',
     'E1: industry requirements, project 3. E2: source repository and commit history. E3: team project plan.',408,26)

clear(52,486,520,267)
table(['Milestone / deliverable','Planned date','Current status / %','Actual result and evidence'],[
    ['Project start and initial coding prototype','03 Sep 2026 (estimate)','Initial coding completed','05 Sep 2026: repository created and initial coding committed to GitHub; E2.'],
    ['Receive initial company image dataset','07 Sep 2026 (estimate)','Received','09 Sep 2026: initial photos received from the company, as confirmed by the team.'],
    ['Develop image-upload user interface','14 Sep 2026 (estimate)','Initial UI implemented','16 Sep 2026: initial user interface committed to GitHub; E2.'],
    ['Train and deploy initial image-based model','23 Sep 2026 (estimate)','Prototype implemented; accuracy validation in progress','25 Sep 2026: machine-learning implementation committed to GitHub. CVAT-labelled company images used for initial training; E2.'],
], [121,64,119,178],490,size=7.8)
para('Overall progress versus initial plan:',
     'Planned dates above are retrospective estimates, not recorded commitments. Development began on 05 Sep 2026 and reached the image-based model. Further development and live-system validation are blocked until company footage and images arrive. No USB handover date or project deadline has been agreed.',704,47,size=8.5)
c.showPage()

page(3)
clear(52,116,493,145)
para('Technologies, tools, standards or methods used and justification:',
     'Python provides a reusable library interface; OpenCV handles images; Ultralytics YOLO learns droplet appearance from labelled examples; CVAT supplies bounding-box labels; Streamlit provides the upload and results dashboard (E2).',120,43)
para('Current implementation / integration status:',
     'Local image inference and model loading are implemented. Current output includes droplet number, image coordinates and confidence. Live USB capture, start/stop control, event timing, recording and hydrotester integration are not implemented in the inspected prototype.',168,45)
para('Technical challenges or decisions made this period:',
     'Reference differencing was sensitive to texture and image mismatch. The model approach removes the reference requirement, but missed droplets persist. Data diversity, missing labels, small droplets and confidence settings need evaluation.',217,39)

clear(52,305,520,276)
table(['Test / validation activity','Result / measurement','Baseline / target','Evidence ID'],[
    ['Dataset and label inventory','50 training images / 30 label files; 8 validation images / 4 label files. No-label images need review.','Audit every image; include genuine dry negatives.','Local dataset'],
    ['Saved training log, epoch 32','Precision 1.000; recall 0.98691; mAP50 0.995; mAP50-95 0.59979. These are logged validation metrics only.','Independent held-out testing across at least three fabrics.','Local results.csv'],
    ['Single-photo and live acceptance tests','User reported missed droplets, including training photos. Live timing and unique-leak tests not yet demonstrated.','First three separate positions; timing tolerance to be agreed.','E2 / team trial logs pending'],
], [128,173,120,61],309,size=7.6)
para('Stakeholder/company feedback obtained and how it changed the work:',
     'The company advised that machine development had delayed data collection and that the machine was completed only recently. Operating footage and more images were promised on a USB stick but remain undelivered. We therefore remain limited to the initial company photos; further development depends on receiving this data.',493,47,size=8.3)
para('Outstanding validation required:',
     'Company footage is required to validate first-three-droplet tracking, detection times, false positives and missed detections under actual machine conditions, including vibration, lighting changes and growing droplets.',548,31,size=8.3)

clear(57,628,488,123)
para('Comparable existing method / product / process:',
     'The earlier project prototype compares a dry reference with a test image. The company\'s current operational procedure and any commercial comparator require confirmation.',632,36)
para('What is different, and evidence of benefit:',
     'The new detector accepts one fabric image without requiring a matched dry reference. The reusable Python function can form the detector stage of a future video pipeline. Reduced setup effort is a design benefit; accuracy, cost and speed gains have not yet been established by controlled comparison.',679,65)
c.showPage()

page(4)
clear(52,100,493,268)
table(['Next activity / deliverable','Owner','Target date','Dependency / success measure'],[
    ['Obtain company operating footage and additional images on the promised USB stick.','Company representatives / Bihara','Company USB handover to Bihara; date unconfirmed','Company representatives will hand the USB containing the required information to Bihara. Essential data is not yet received.'],
    ['Label new data, improve the model and evaluate unseen images and video.','All team members','After USB handover to Bihara','Blocked until company data arrives. Measure false positives and misses across at least three fabrics.'],
    ['Complete live capture, independent-leak tracking, timing, recording and application integration.','All team members','After data receipt and model evaluation','Representative machine footage is required for development and validation. Overall delivery date not agreed.'],
], [152,59,61,210],103,size=7.5)
table(['Risk / blocker','Impact & likelihood','Mitigation / support needed','Owner'],[
    ['Required company footage and images not supplied','High impact; delay ongoing','Company must supply operating data. Further development is blocked until receipt; no handover date agreed.','Team / company'],
    ['Insufficient data for final accuracy and live validation','High impact; high likelihood while data is absent','Obtain varied operating examples; label, retrain and validate detection timing and separate leak positions.','All members'],
], [121,94,208,59],278,size=7.3)

clear(52,405,493,346)
text('Evidence supporting the requirements, completed development and planned work is listed below. Training measurements are recorded in runs/detect/train-2/results.csv; image and label files are stored in datasets/.',57,409,482,34,8.6)
table(['Evidence ID','Description and date','Accessible link / file name'],[
    ['E1','Industry proposal: project 3, page 4; water-leak detection requirements.','TransitionX_2026_Industry_Project_Proposals.pdf'],
    ['E2','Source code committed to GitHub, with commit messages recording development progress. Started 05 Sep; UI 16 Sep; machine learning 25 Sep 2026 (Sri Lanka time).','https://github.com/IT24103447/droplet-detector'],
    ['E3','Team project plan; supplied planning-document link.','https://drive.google.com/file/d/1czQJlwkYZfaBEJ1bS5s6wZEb0zNQhblT/view?usp=sharing'],
], [60,181,241],450,size=8)
c.showPage()
c.save()
print(OUT)
