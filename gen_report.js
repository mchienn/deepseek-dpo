const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, PageBreak, Header, Footer
} = require('docx');
const fs = require('fs');
const path = require('path');

const NAVY="1B3A6B",BLUE="2563EB",BLUE_L="DBEAFE",
      TEAL="0F6E56",TEAL_L="D1FAE5",AMBER="854F0B",AMBER_L="FEF3C7",
      RED="991B1B",RED_L="FEE2E2",GRAY="374151",GRAY_L="F9FAFB",
      GREEN="166534",GREEN_L="DCFCE7",PURPLE="534AB7",PURPLE_L="EEEDFE",
      BORDER="D1D5DB",WHITE="FFFFFF",MONO_BG="F1F5F9",MONO_FG="1E293B";

const bdr=(c=BORDER)=>({style:BorderStyle.SINGLE,size:1,color:c});
const bdrs=(c=BORDER)=>({top:bdr(c),bottom:bdr(c),left:bdr(c),right:bdr(c)});
const CM={top:80,bottom:80,left:120,right:120};
const sp=(b=0,a=0)=>({spacing:{before:b,after:a}});

const R=(t,o={})=>new TextRun({text:t,font:"Arial",size:21,...o});
const M=t=>new TextRun({text:t,font:"Courier New",size:19,color:MONO_FG});
const H1=t=>new Paragraph({heading:HeadingLevel.HEADING_1,...sp(320,140),
  border:{bottom:{style:BorderStyle.SINGLE,size:6,color:BLUE,space:6}},
  children:[R(t,{bold:true,size:30,color:NAVY})]});
const H2=t=>new Paragraph({heading:HeadingLevel.HEADING_2,...sp(240,100),
  children:[R(t,{bold:true,size:24,color:NAVY})]});
const H3=t=>new Paragraph({heading:HeadingLevel.HEADING_3,...sp(180,70),
  children:[R(t,{bold:true,size:21,color:GRAY})]});
const P=(t,o={})=>new Paragraph({...sp(50,70),children:[R(t,o)]});
const Gap=()=>new Paragraph({...sp(60,60),children:[]});
const PB=()=>new Paragraph({children:[new PageBreak()]});
const Bullet=t=>new Paragraph({numbering:{reference:"bullets",level:0},...sp(35,35),children:[R(t)]});
const Sub=t=>new Paragraph({numbering:{reference:"sub",level:0},...sp(25,25),children:[R(t,{color:GRAY})]});
const Note=(t,color=BLUE)=>new Paragraph({...sp(100,100),indent:{left:320},
  border:{left:{style:BorderStyle.THICK,size:14,color,space:7}},
  children:[R(t,{italics:true,color:GRAY})]});
const Code=lines=>lines.map(l=>new Paragraph({...sp(0,2),
  shading:{fill:MONO_BG,type:ShadingType.CLEAR},indent:{left:320},children:[M(l)]}));
const HC=(t,fill=NAVY,w=1800)=>new TableCell({borders:bdrs(fill),width:{size:w,type:WidthType.DXA},
  shading:{fill,type:ShadingType.CLEAR},margins:CM,
  children:[new Paragraph({children:[R(t,{bold:true,size:19,color:WHITE})]})]});
const PC=(t,fill=WHITE,w=1800,opts={})=>new TableCell({borders:bdrs(BORDER),
  width:{size:w,type:WidthType.DXA},shading:{fill,type:ShadingType.CLEAR},margins:CM,
  children:Array.isArray(t)?[new Paragraph({children:t})]
    :[new Paragraph({children:[R(t,{size:19,...opts})]})]});
const row=cells=>new TableRow({children:cells});

const LIGHT_FILLS={[TEAL]:TEAL_L,[RED]:RED_L,[AMBER]:AMBER_L,[BLUE]:BLUE_L};
const vBadge=(ver,fill,textColor,desc)=>new Table({
  width:{size:9000,type:WidthType.DXA},columnWidths:[1200,7800],
  rows:[row([
    new TableCell({borders:bdrs(fill),width:{size:1200,type:WidthType.DXA},
      shading:{fill,type:ShadingType.CLEAR},margins:CM,
      children:[new Paragraph({alignment:AlignmentType.CENTER,
        children:[R(ver,{bold:true,size:24,color:textColor})]})]}),
    new TableCell({borders:bdrs(fill),width:{size:7800,type:WidthType.DXA},
      shading:{fill:LIGHT_FILLS[fill]||GRAY_L,type:ShadingType.CLEAR},margins:CM,
      children:[new Paragraph({children:[R(desc,{size:20,color:GRAY})]})]}),
  ])]
});

const doc=new Document({
  numbering:{config:[
    {reference:"bullets",levels:[{level:0,format:LevelFormat.BULLET,text:"â€¢",
      alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:680,hanging:340}}}}]},
    {reference:"sub",levels:[{level:0,format:LevelFormat.BULLET,text:"â—¦",
      alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:1020,hanging:340}}}}]},
  ]},
  styles:{default:{document:{run:{font:"Arial",size:21}}},
    paragraphStyles:[
      {id:"Heading1",name:"Heading 1",basedOn:"Normal",next:"Normal",quickFormat:true,
        run:{size:30,bold:true,font:"Arial",color:NAVY},
        paragraph:{spacing:{before:320,after:140},outlineLevel:0}},
      {id:"Heading2",name:"Heading 2",basedOn:"Normal",next:"Normal",quickFormat:true,
        run:{size:24,bold:true,font:"Arial",color:NAVY},
        paragraph:{spacing:{before:240,after:100},outlineLevel:1}},
      {id:"Heading3",name:"Heading 3",basedOn:"Normal",next:"Normal",quickFormat:true,
        run:{size:21,bold:true,font:"Arial",color:GRAY},
        paragraph:{spacing:{before:180,after:70},outlineLevel:2}},
    ]},
  sections:[{
    properties:{page:{size:{width:12240,height:15840},margin:{top:1300,right:1200,bottom:1300,left:1200}}},
    headers:{default:new Header({children:[new Paragraph({
      border:{bottom:{style:BorderStyle.SINGLE,size:4,color:BLUE,space:4}},...sp(0,100),
      children:[R("DeepSeek-R1-Distill-Qwen-1.5B | Web Automation | Version History & Analysis",{size:17,color:"9CA3AF"})]
    })]})},
    footers:{default:new Footer({children:[new Paragraph({
      border:{top:{style:BorderStyle.SINGLE,size:4,color:BORDER,space:4}},...sp(100,0),
      children:[R("Trang ",{size:17,color:"9CA3AF"}),
        new TextRun({children:[PageNumber.CURRENT],font:"Arial",size:17,color:"9CA3AF"}),
        R(" / ",{size:17,color:"9CA3AF"}),
        new TextRun({children:[PageNumber.TOTAL_PAGES],font:"Arial",size:17,color:"9CA3AF"})]
    })]})},
    children:[

      new Paragraph({alignment:AlignmentType.CENTER,...sp(1200,80),
        children:[R("MODEL VERSION HISTORY",{bold:true,size:48,color:NAVY})]}),
      new Paragraph({alignment:AlignmentType.CENTER,...sp(0,80),
        children:[R("DeepSeek-R1-Distill-Qwen-1.5B â€” Web Automation JSON Generation",{size:26,color:GRAY})]}),
      new Paragraph({alignment:AlignmentType.CENTER,...sp(0,1200),
        children:[R("PhÃ¢n tÃ­ch toÃ n diá»‡n SFT â†’ DPO v1 â†’ DPO v2 | Root cause & Roadmap",{size:20,italics:true,color:"9CA3AF"})]}),
      new Paragraph({...sp(0,1200),alignment:AlignmentType.CENTER,
        border:{bottom:{style:BorderStyle.SINGLE,size:8,color:BLUE,space:1}},children:[]}),
      PB(),

      H1("1. Version History â€” Báº£ng so sÃ¡nh tá»•ng quan"),
      Gap(),
      new Table({width:{size:9200,type:WidthType.DXA},columnWidths:[1400,1600,1600,1600,1600,1400],
        rows:[
          row([HC("Version",NAVY,1400),HC("Total",NAVY,1600),HC("Pass Rate",NAVY,1600),
               HC("Mean Score",NAVY,1600),HC("Invalid JSON",NAVY,1600),HC("Score =1.0",NAVY,1400)]),
          row([PC("v1.0 â€” SFT",GRAY_L,1400,{bold:true}),
               PC("10,050",GRAY_L,1600),
               PC("91.04%",TEAL_L,1600,{bold:true,color:TEAL}),
               PC("0.9005",TEAL_L,1600,{bold:true,color:TEAL}),
               PC("43",TEAL_L,1600,{bold:true,color:TEAL}),
               PC("3,863",TEAL_L,1400,{bold:true,color:TEAL})]),
          row([PC("v2.0 â€” DPO buggy",RED_L,1400,{bold:true}),
               PC("9,427 âš ",RED_L,1600,{color:RED}),
               PC("90.47% â†“",RED_L,1600,{bold:true,color:RED}),
               PC("0.8945 â†“",RED_L,1600,{bold:true,color:RED}),
               PC("69 â†‘",RED_L,1600,{bold:true,color:RED}),
               PC("3,246 â†“",RED_L,1400,{bold:true,color:RED})]),
          row([PC("v2.1 â€” DPO buggy*",AMBER_L,1400,{bold:true}),
               PC("9,427 âš ",AMBER_L,1600,{color:AMBER}),
               PC("90.47% (=v2.0)",AMBER_L,1600,{color:AMBER}),
               PC("0.8945 (=v2.0)",AMBER_L,1600,{color:AMBER}),
               PC("69",AMBER_L,1600,{color:AMBER}),
               PC("3,290",AMBER_L,1400,{color:AMBER})]),
          row([PC("v3.0 â€” DPO v2 target",BLUE_L,1400,{bold:true}),
               PC("10,050 âœ“",BLUE_L,1600,{color:BLUE}),
               PC("> 92%",BLUE_L,1600,{bold:true,color:BLUE}),
               PC("> 0.91",BLUE_L,1600,{bold:true,color:BLUE}),
               PC("< 25",BLUE_L,1600,{bold:true,color:BLUE}),
               PC("> 4,000",BLUE_L,1400,{bold:true,color:BLUE})]),
        ]}),
      Gap(),
      Note("* v2.1 lÃ  eval má»›i nháº¥t Ä‘Æ°á»£c upload â€” sá»‘ liá»‡u GIá»NG Há»†T v2.0 Ä‘áº¿n 6 chá»¯ sá»‘ tháº­p phÃ¢n. ÄÃ¢y lÃ  cÃ¹ng 1 eval file, khÃ´ng pháº£i DPO v2 Ä‘Ã£ fix.", AMBER),
      Gap(),

      H2("1.1. Äá»‹nh nghÄ©a cÃ¡c version"),
      Gap(),
      vBadge("v1.0",TEAL,WHITE,"SFT baseline â€” QLoRA fine-tune 6,000 samples, 3 phase curriculum, checkpoint-200, test 10,050 samples."),
      Gap(),
      vBadge("v2.0",RED,WHITE,"DPO train Ä‘áº§u tiÃªn â€” 5 bugs: MAX_LEN=512, wrap_pair leak, no ternary corruption, epochs=3, beta=0.1. Eval chá»‰ 9,427 samples."),
      Gap(),
      vBadge("v2.1",AMBER,WHITE,"DPO sau khi fix bugs trong code â€” nhÆ°ng eval_deepseek_result.xlsx upload lÃªn lÃ  SAME FILE vá»›i v2.0, khÃ´ng pháº£i káº¿t quáº£ cháº¡y má»›i."),
      Gap(),
      vBadge("v3.0",BLUE,WHITE,"Target: DPO v2 Ä‘Ã£ fix + data pipeline cáº£i thiá»‡n. Xem Part 3 & 4."),
      Gap(),

      PB(),

      H1("2. Root Cause Analysis â€” Táº¡i sao DPO khÃ´ng cáº£i thiá»‡n"),
      Gap(),
      P("PhÃ¢n tÃ­ch 898 fail cases tá»« v2.x. Chia thÃ nh 4 nhÃ³m nguyÃªn nhÃ¢n:"),
      Gap(),

      new Table({width:{size:9200,type:WidthType.DXA},columnWidths:[1200,1200,1600,5200],
        rows:[
          row([HC("NhÃ³m",NAVY,1200),HC("Count",NAVY,1200),HC("% of Fails",NAVY,1600),HC("NguyÃªn nhÃ¢n",NAVY,5200)]),
          row([PC("Cat 1",RED_L,1200,{bold:true,color:RED}),PC("69",RED_L,1200),
               PC("7.7%",RED_L,1600),PC("Invalid JSON â€” trailing comma, truncation, reasoning leak",WHITE,5200)]),
          row([PC("Cat 2",AMBER_L,1200,{bold:true,color:AMBER}),PC("152",AMBER_L,1200),
               PC("16.9%",AMBER_L,1600),PC("Action mismatch â€” OOD actions (play, submit, load, keyboard, set) khÃ´ng cÃ³ trong canonical set",WHITE,5200)]),
          row([PC("Cat 3",PURPLE_L,1200,{bold:true,color:PURPLE}),PC("677",PURPLE_L,1200),
               PC("75.4%",PURPLE_L,1600),PC("Score < 0.7: action Ä‘Ãºng nhÆ°ng expected keys sai hoáº·c selector value mismatch",WHITE,5200)]),
        ]}),
      Gap(),

      H3("Cat 3 â€” Deep dive: 75% fail cases Ä‘áº¿n tá»« Ä‘Ã¢y"),
      P("ÄÃ¢y lÃ  nhÃ³m quan trá»ng nháº¥t. Trong 677 cases action Ä‘Ãºng nhÆ°ng score < 0.7:"),
      Gap(),
      Bullet("570/677 (84%) bá»‹ key mismatch < 50% overlap giá»¯a gold vÃ  model expected"),
      Bullet("Missed keys phá»• biáº¿n nháº¥t: text-content (82), border-bottom-color (51), border-bottom-width (50), border-bottom-style (49), visibility (39)"),
      Bullet("Extra keys model thÃªm khÃ´ng cÃ³ trong gold: visibility (113), text-content (77), count (55), width (44), background-color (35)"),
      Bullet("54 cases gold selector rá»—ng nhÆ°ng model Ä‘iá»n selector vÃ o â†’ penalty score"),
      Gap(),
      P("Diá»…n giáº£i: model Ä‘ang 'Ä‘oÃ¡n thÃªm' properties khÃ´ng Ä‘Æ°á»£c yÃªu cáº§u (hallucination props), Ä‘á»“ng thá»i bá» sÃ³t cÃ¡c properties Ä‘Æ°á»£c yÃªu cáº§u. ÄÃ¢y lÃ  váº¥n Ä‘á» KHÃ”NG Ä‘Æ°á»£c giáº£i quyáº¿t bá»Ÿi DPO pairs hiá»‡n táº¡i vÃ¬ synthetic corruption chá»‰ corrupt/drop keys, khÃ´ng dáº¡y model khi nÃ o KHÃ”NG nÃªn thÃªm key má»›i."),
      Gap(),

      H3("Cat 2 â€” OOD actions: 152 cases"),
      Gap(),
      new Table({width:{size:9200,type:WidthType.DXA},columnWidths:[2200,2200,1400,3400],
        rows:[
          row([HC("Gold action",NAVY,2200),HC("Model predicted",NAVY,2200),HC("Count",NAVY,1400),HC("Fix",NAVY,3400)]),
          ...[
            ["verify","click","22","Ambiguous cmd â€” thÃªm training samples verify vs click"],
            ["goto","click","21","goto bá»‹ confuse vá»›i click â€” thÃªm goto samples rÃµ rÃ ng hÆ¡n"],
            ["submit","click","6","submit â†’ map sang click trong canonical"],
            ["load","goto","3","load â†’ map sang goto trong canonical"],
            ["play","goto","3","play khÃ´ng cÃ³ trong canonical â€” thÃªm hoáº·c map"],
            ["keyboard","click","2","keyboard khÃ´ng cÃ³ canonical â€” map sang click/input"],
            ["set","scroll","2","set khÃ´ng canonical â€” map sang input"],
          ].map(([g,p,c,f],i)=>row([
            PC(g,i%2===0?GRAY_L:WHITE,2200),PC(p,i%2===0?GRAY_L:WHITE,2200),
            PC(c,i%2===0?GRAY_L:WHITE,1400),PC(f,i%2===0?GRAY_L:WHITE,3400)
          ]))
        ]}),
      Gap(),
      Note("Root cause: build_dpo_pairs.py cÃ³ CANONICAL_TO_WRONG_KEY nhÆ°ng KHÃ”NG cÃ³ canonical action mapping cho OOD actions. Model gáº·p 'play', 'submit', 'load', 'keyboard', 'set' trong test â€” khÃ´ng biáº¿t map sang gÃ¬.", RED),
      Gap(),

      PB(),

      H1("3. Káº¿t luáº­n: DPO Ä‘Ãºng hÆ°á»›ng nhÆ°ng data pairs chÆ°a target Ä‘Ãºng váº¥n Ä‘á»"),
      Gap(),
      P("DPO hiá»‡n táº¡i chá»‰ fix Ä‘Æ°á»£c Cat 1 (invalid JSON) má»™t pháº§n. Cat 2 vÃ  Cat 3 chiáº¿m 92.3% fail cases nhÆ°ng KHÃ”NG cÃ³ pair nÃ o trong dpo_pairs.jsonl address trá»±c tiáº¿p hai váº¥n Ä‘á» nÃ y:"),
      Gap(),

      new Table({width:{size:9200,type:WidthType.DXA},columnWidths:[3000,1600,1600,2000,1000],
        rows:[
          row([HC("Váº¥n Ä‘á»",NAVY,3000),HC("Fail count",NAVY,1600),HC("DPO pairs hiá»‡n cÃ³",NAVY,1600),HC("DPO pairs cáº§n thÃªm",NAVY,2000),HC("Fix",NAVY,1000)]),
          row([PC("Invalid JSON (Cat 1)",WHITE,3000),PC("69",WHITE,1600),
               PC("âœ“ cÃ³ (invalid_json source)",TEAL_L,1600,{color:TEAL}),
               PC("Äá»§",WHITE,2000),PC("âœ“",TEAL_L,1000,{bold:true,color:TEAL})]),
          row([PC("OOD action mismatch (Cat 2)",GRAY_L,3000),PC("152",GRAY_L,1600),
               PC("âœ— khÃ´ng cÃ³",RED_L,1600,{color:RED}),
               PC("pairs vá»›i gold=submit/load/play â†’ chosen=click/goto",GRAY_L,2000),
               PC("âœ—",RED_L,1000,{bold:true,color:RED})]),
          row([PC("Key hallucination â€” thÃªm key khÃ´ng cáº§n (Cat 3)",WHITE,3000),PC("~300",WHITE,1600),
               PC("âœ— khÃ´ng cÃ³",RED_L,1600,{color:RED}),
               PC("pairs vá»›i rejected=extra keys, chosen=minimal correct",WHITE,2000),
               PC("âœ—",RED_L,1000,{bold:true,color:RED})]),
          row([PC("Key miss â€” thiáº¿u key Ä‘Æ°á»£c yÃªu cáº§u (Cat 3)",GRAY_L,3000),PC("~370",GRAY_L,1600),
               PC("Partial (drop_key corrupt)",AMBER_L,1600,{color:AMBER}),
               PC("pairs vá»›i rejected=missing border-* keys, chosen=full",GRAY_L,2000),
               PC("â–³",AMBER_L,1000,{bold:true,color:AMBER})]),
          row([PC("Selector value mismatch (Cat 3)",WHITE,3000),PC("54",WHITE,1600),
               PC("âœ— khÃ´ng cÃ³",RED_L,1600,{color:RED}),
               PC("pairs vá»›i gold sel='', rejected sel=cÃ³ giÃ¡ trá»‹",WHITE,2000),
               PC("âœ—",RED_L,1000,{bold:true,color:RED})]),
        ]}),
      Gap(),

      PB(),

      H1("4. Roadmap cáº£i thiá»‡n â€” v3.0"),
      Gap(),

      H2("4.1. Fix build_dpo_pairs.py â€” thÃªm 3 nguá»“n pairs má»›i"),
      Gap(),

      H3("Nguá»“n má»›i 1: OOD action pairs"),
      P("ThÃªm function extract_ood_action_pairs() â€” khai thÃ¡c Ä‘Ãºng 152 cases Cat 2 Ä‘Ã£ cÃ³ trong eval:"),
      ...Code([
        "# ThÃªm vÃ o CANONICAL_ACTION_MAP:",
        "OOD_TO_CANONICAL = {",
        "    'submit': 'click', 'load': 'goto', 'play': 'click',",
        "    'keyboard': 'input', 'set': 'input', 'type': 'input',",
        "    'open': 'goto', 'check': 'verify', 'validate': 'verify',",
        "}",
        "",
        "def extract_ood_action_pairs(eval_df):",
        "    pairs = []",
        "    for _, row in eval_df.iterrows():",
        "        gold = safe_json_loads(row['Step Object'])",
        "        if not isinstance(gold, dict): continue",
        "        action = gold.get('action', '')",
        "        if action not in OOD_TO_CANONICAL: continue",
        "        # chosen = same JSON vá»›i canonical action",
        "        fixed = dict(gold)",
        "        fixed['action'] = OOD_TO_CANONICAL[action]",
        "        pairs.append({",
        "            'source': f'ood_action_{action}',",
        "            'prompt': row['Sub Task'],",
        "            'chosen': json.dumps(fixed),",
        "            'rejected': row['Step Object'],  # original vá»›i OOD action",
        "        })",
        "    return pairs",
      ]),
      Gap(),

      H3("Nguá»“n má»›i 2: Key hallucination pairs"),
      P("Dáº¡y model KHÃ”NG thÃªm keys khÃ´ng Ä‘Æ°á»£c yÃªu cáº§u â€” láº¥y tá»« eval Cat 3 cases:"),
      ...Code([
        "def extract_key_hallucination_pairs(eval_df):",
        "    pairs = []",
        "    for _, row in eval_df.iterrows():",
        "        if row['AVG Score'] >= 0.7: continue",
        "        gold = safe_json_loads(row['Step Object'])",
        "        pred = safe_json_loads(row['Model'])",
        "        if not isinstance(gold, dict) or not isinstance(pred, dict): continue",
        "        if gold.get('action') != pred.get('action'): continue",
        "        gold_keys = set(gold.get('expected', {}).keys())",
        "        pred_keys = set(pred.get('expected', {}).keys())",
        "        extra = pred_keys - gold_keys  # keys pred thÃªm mÃ  gold khÃ´ng cÃ³",
        "        if not extra: continue",
        "        # chosen = gold (minimal, khÃ´ng cÃ³ extra keys)",
        "        # rejected = pred (cÃ³ extra hallucinated keys)",
        "        pairs.append({",
        "            'source': 'key_hallucination',",
        "            'prompt': row['Sub Task'],",
        "            'chosen': row['Step Object'],",
        "            'rejected': row['Model'],",
        "        })",
        "    return pairs",
      ]),
      Gap(),

      H3("Nguá»“n má»›i 3: Selector empty pairs"),
      P("Dáº¡y model khÃ´ng Ä‘iá»n selector khi gold Ä‘á»ƒ trá»‘ng:"),
      ...Code([
        "def extract_empty_selector_pairs(eval_df):",
        "    pairs = []",
        "    for _, row in eval_df.iterrows():",
        "        gold = safe_json_loads(row['Step Object'])",
        "        pred = safe_json_loads(row['Model'])",
        "        if not isinstance(gold, dict) or not isinstance(pred, dict): continue",
        "        if gold.get('selector', '') != '': continue  # gold pháº£i empty",
        "        if pred.get('selector', '') == '': continue  # pred pháº£i NON-empty (lá»—i)",
        "        pairs.append({",
        "            'source': 'empty_selector',",
        "            'prompt': row['Sub Task'],",
        "            'chosen': row['Step Object'],   # empty selector = Ä‘Ãºng",
        "            'rejected': row['Model'],        # non-empty selector = sai",
        "        })",
        "    return pairs",
      ]),
      Gap(),

      H2("4.2. Fix train_dpo.py â€” cÃ¡c thay Ä‘á»•i cho v3.0"),
      Gap(),
      new Table({width:{size:9200,type:WidthType.DXA},columnWidths:[2200,2000,2000,3000],
        rows:[
          row([HC("Parameter",NAVY,2200),HC("v2.x (buggy)",NAVY,2000),HC("v3.0",NAVY,2000),HC("LÃ½ do",NAVY,3000)]),
          ...[
            ["MAX_LEN","512","1024","Prevent completion truncation"],
            ["num_train_epochs","3","2","Prevent overfitting vá»›i dataset nhá»"],
            ["beta","0.1","0.2","Stronger KL penalty â†’ giá»¯ gáº§n SFT base"],
            ["wrap_pair()","CÃ“ (leak)","XÃ“A","Reasoning trong chosen/rejected gÃ¢y leak"],
            ["DPO pair sources","4 nguá»“n","7 nguá»“n","ThÃªm OOD action, key hallucination, empty selector"],
            ["MIN_PAIRS_TARGET","2,000","3,500","Nhiá»u pairs hÆ¡n cho 3 nguá»“n lá»—i má»›i"],
          ].map(([p,b,v,r],i)=>row([
            PC(p,i%2===0?GRAY_L:WHITE,2200,{bold:true}),
            PC(b,i%2===0?RED_L:WHITE,2000,{color:RED}),
            PC(v,i%2===0?TEAL_L:WHITE,2000,{color:TEAL,bold:true}),
            PC(r,i%2===0?GRAY_L:WHITE,3000)
          ]))
        ]}),
      Gap(),

      H2("4.3. Cáº£i thiá»‡n eval script â€” fix 622 missing samples"),
      ...Code([
        "# eval_deepseek.py â€” thÃªm try/except per sample Ä‘á»ƒ khÃ´ng bá»‹ skip",
        "MAX_NEW_TOKENS = 512   # tÄƒng tá»« 256 Ä‘á»ƒ cover samples dÃ i hÆ¡n",
        "",
        "for b in tqdm(range(num_batches)):",
        "    batch = remaining[b*HF_BATCH:(b+1)*HF_BATCH]",
        "    try:",
        "        inputs = tokenizer(prompts, ..., max_length=512)  # tÄƒng tá»« 256",
        "        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)",
        "    except RuntimeError as e:",
        "        if 'out of memory' in str(e).lower():",
        "            for idx, subtask in zip(batch_idxs, batch_subtasks):",
        "                try:",
        "                    # single sample inference",
        "                    ...",
        "                except:",
        "                    done[idx] = ('{\"action\":\"error\"}', 'OOM')",
        "        continue",
      ]),
      Gap(),

      H2("4.4. Dá»± bÃ¡o cáº£i thiá»‡n sau v3.0"),
      Gap(),
      new Table({width:{size:9200,type:WidthType.DXA},columnWidths:[3000,1600,1600,3000],
        rows:[
          row([HC("Metric",NAVY,3000),HC("v1.0 SFT",NAVY,1600),HC("v3.0 target",NAVY,1600),HC("Nguá»“n cáº£i thiá»‡n",NAVY,3000)]),
          ...[
            ["Pass rate","91.04%","> 93%","Giáº£i quyáº¿t Cat 2+3"],
            ["Score = 1.0","3,863","> 4,200","Ãt key hallucination hÆ¡n"],
            ["Invalid JSON","43","< 20","MAX_LEN fix + no wrap_pair"],
            ["OOD action accuracy","~60%","> 85%","OOD action pairs má»›i"],
            ["Key precision","~70%","> 82%","Key hallucination pairs"],
            ["Test samples covered","10,050","10,050","eval script fix"],
          ].map(([m,s,t,r],i)=>row([
            PC(m,i%2===0?GRAY_L:WHITE,3000,{bold:true}),
            PC(s,i%2===0?GRAY_L:WHITE,1600),
            PC(t,i%2===0?TEAL_L:WHITE,1600,{bold:true,color:TEAL}),
            PC(r,i%2===0?GRAY_L:WHITE,3000)
          ]))
        ]}),
      Gap(),

      H2("4.5. Quyáº¿t Ä‘á»‹nh: tiáº¿p tá»¥c DPO hay chuyá»ƒn sang GRPO?"),
      Gap(),
      P("Dá»±a trÃªn phÃ¢n tÃ­ch, DPO CHÆ¯A lÃ  bottleneck â€” váº¥n Ä‘á» lÃ  DPO pairs chÆ°a cover Ä‘Ãºng fail cases. Tráº£ lá»i cÃ¢u há»i nÃ y theo Ä‘iá»u kiá»‡n:"),
      Gap(),
      Bullet("Náº¿u sau v3.0 pass rate > 93% â†’ DPO Ä‘á»§ tá»‘t, khÃ´ng cáº§n GRPO"),
      Bullet("Náº¿u Cat 3 (key hallucination) váº«n chiáº¿m > 50% fails sau v3.0 â†’ Ä‘Ã¢y lÃ  váº¥n Ä‘á» SFT data quality, khÃ´ng pháº£i GRPO giáº£i quyáº¿t Ä‘Æ°á»£c"),
      Bullet("GRPO chá»‰ nÃªn dÃ¹ng khi: model Ä‘Ã£ há»c Ä‘Ãºng pattern nhÆ°ng cáº§n fine-tune boundary cases â€” hiá»‡n táº¡i model chÆ°a há»c Ä‘á»§ domain knowledge Ä‘á»ƒ GRPO cÃ³ signal tá»‘t"),
      Gap(),
      Note("Thá»© tá»± Æ°u tiÃªn: (1) Fix eval script Ä‘á»ƒ cÃ³ 10,050 samples apples-to-apples. (2) Cháº¡y DPO v3.0 vá»›i 3 nguá»“n pairs má»›i. (3) Äo láº¡i. Chá»‰ sau Ä‘Ã³ má»›i quyáº¿t Ä‘á»‹nh GRPO.", BLUE),
      Gap(),

    ]
  }]
});

const docsDir = path.join(__dirname, 'docs', 'history');
fs.mkdirSync(docsDir, { recursive: true });
const outPath = path.join(docsDir, 'DPO_Version_History_Report.docx');
Packer.toBuffer(doc).then(buf=>{
  fs.writeFileSync(outPath, buf);
  console.log("Done -> " + outPath);
});
