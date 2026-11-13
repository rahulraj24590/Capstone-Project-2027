"""
Multilingual Safety Guidance Module (Phase 8).

Provides authentic, verified landslide safety protocols in English, Hindi, and Punjabi,
strictly following NDMA (National Disaster Management Authority, India), GSI (Geological
Survey of India), and USGS civil-defense guidelines.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "en": {"code": "en", "label": "English", "native": "English"},
    "hi": {"code": "hi", "label": "Hindi", "native": "हिन्दी"},
    "pa": {"code": "pa", "label": "Punjabi", "native": "ਪੰਜਾਬੀ"},
}

SAFETY_DATABASE: dict[str, dict[str, Any]] = {
    "en": {
        "language": "en",
        "title": "Landslide Preparedness & Emergency Safety Guide",
        "authority": "National Disaster Management Authority (NDMA) & Geological Survey of India (GSI)",
        "summary": "Verified safety protocols for mountainous and hilly terrains prone to rainfall-triggered landslides and debris flows.",
        "before": {
            "category": "Before a Landslide (Preparedness & Early Warning)",
            "warning_signs_heading": "Recognize Warning Signs",
            "warning_signs": [
                "New cracks appearing in plaster, tile, brickwork, or building foundations.",
                "Doors or windows sticking or jamming for the first time.",
                "Outside walls, fences, utility poles, or retaining walls tilting or leaning downslope.",
                "Slowly developing tension cracks appearing on slopes, roads, or ground surfaces.",
                "Water breaking through the ground surface in new locations or unexpected springs emerging.",
                "Sudden muddiness or dramatic decrease/increase in water levels of local streams and drainage channels.",
                "Faint rumbling sounds that gradually increase in volume as a slide approaches.",
            ],
            "action_items_heading": "Protective Measures",
            "action_items": [
                "Identify multiple evacuation routes away from steep slopes, stream channels, and ravines.",
                "Prepare an Emergency Grab Kit: first aid supplies, torch/flashlight, battery radio, non-perishable food, potable water, essential medications, and identification documents in a waterproof pouch.",
                "Keep emergency phone numbers (112, 1078, 1070) saved and accessible on all family mobile devices.",
                "Monitor official weather warnings from IMD and local district administration bulletins during heavy monsoon spells.",
            ],
        },
        "during": {
            "category": "During a Landslide (Immediate Emergency Actions)",
            "indoor_heading": "If Indoors",
            "indoor_actions": [
                "Quickly move to a higher floor or upper level of the structure away from downslope walls.",
                "Take cover under a sturdy piece of furniture (heavy table or bench) to shield against collapsing debris.",
                "Stay away from windows and downslope-facing glass panes.",
            ],
            "outdoor_heading": "If Outdoors",
            "outdoor_actions": [
                "Quickly move away from the path of the slide, debris flow, or rockfall.",
                "Run laterally or upslope toward stable bedrock, ridges, or wide elevated ground.",
                "Avoid low-lying riverbeds, valleys, ravines, and narrow gullies.",
            ],
            "vehicle_heading": "If in a Moving Vehicle",
            "vehicle_actions": [
                "Watch for washed-out pavements, fallen rocks, mud accumulation, and collapsing road shoulders.",
                "Do NOT attempt to cross flooded bridges, culverts, or active debris streams.",
                "Pull over to a safe, stable open shoulder away from overhangs and steep uphill cuts.",
            ],
            "trapped_heading": "If Trapped or Escape is Impossible",
            "trapped_actions": [
                "Curl into a tight ball (fetal position) and protect your head with your arms to reduce the risk of blunt trauma from rolling boulders and debris.",
            ],
        },
        "after": {
            "category": "After a Landslide (Post-Event Safety & Recovery)",
            "safety_heading": "Immediate Safety Rules",
            "safety_rules": [
                "Stay away from the slide area. Secondary slides, back-scarp slumps, and flash floods frequently occur hours or days after the primary collapse.",
                "Check for trapped and injured persons from outside the hazard zone; do NOT enter active debris fields alone.",
                "Direct official emergency rescue teams (NDRF, SDRF, local police) to the locations of trapped individuals.",
                "Report broken utility lines (ruptured gas pipes, severed electrical cables, damaged water mains) immediately to local authorities.",
                "Do NOT return to damaged dwellings until certified safe by local civil engineers or district revenue authorities.",
                "Inspect building foundations, retaining walls, and surrounding land for new fractures before re-entry.",
            ],
        },
        "emergency_contacts": [
            {"service": "National Emergency Helpline", "number": "112", "notes": "All-India 24x7 toll-free emergency response"},
            {"service": "NDMA Disaster Helpline", "number": "1078", "notes": "National Disaster Management Authority (Toll-Free)"},
            {"service": "State Disaster Management (SDMA)", "number": "1070", "notes": "State emergency operation center"},
            {"service": "District Disaster Management (DDMA)", "number": "1077", "notes": "District emergency control room"},
            {"service": "Police Assistance", "number": "100", "notes": "Local police dispatch"},
            {"service": "Medical Ambulance", "number": "108 / 102", "notes": "Emergency medical transport"},
            {"service": "Fire & Rescue Services", "number": "101", "notes": "Fire and collapse rescue"},
        ],
        "disclaimer": "This safety guidance reflects verified civil-defense advisories from NDMA and GSI. It is provided for educational and preparedness purposes. During an active disaster, always follow the direct instructions of local district magistrates, NDRF, and emergency first responders.",
    },
    "hi": {
        "language": "hi",
        "title": "भूस्खलन पूर्व तैयारी एवं आपातकालीन सुरक्षा मार्गदर्शिका",
        "authority": "राष्ट्रीय आपदा प्रबंधन प्राधिकरण (NDMA) एवं भारतीय भूवैज्ञानिक सर्वेक्षण (GSI)",
        "summary": "भारी वर्षा से होने वाले भूस्खलन एवं मलबे के बहाव के प्रति संवेदनशील पहाड़ी क्षेत्रों के लिए प्रमाणित सुरक्षा निर्देश।",
        "before": {
            "category": "भूस्खलन से पहले (पूर्व तैयारी एवं चेतावनी संकेत)",
            "warning_signs_heading": "चेतावनी संकेतों को पहचानें",
            "warning_signs": [
                "दीवारों, फर्श, टाइलों या इमारत की नींव में नई दरारें दिखाई देना।",
                "खिड़कियों या दरवाजों का पहली बार अचानक जाम होना या अटकना।",
                "बाहरी चहारदीवारी, बाड़, बिजली के खंभों या पेड़ों का ढलान की ओर झुकना।",
                "पहाड़ी ढलानों, सड़कों या खुली जमीन पर धीरे-धीरे चौड़ी होती दरारें नजर आना।",
                "जमीन से अप्रत्याशित स्थानों पर अचानक पानी का रिसना या नए झरनों का फूटना।",
                "स्थानीय नालों और जल स्रोतों में अचानक मटमैला पानी बहना या जलस्तर में अचानक उतार-चढ़ाव होना।",
                "दूर से आती गड़गड़ाहट की आवाज का धीरे-धीरे तेज होना, जो मलबे के खिसकने का संकेत है।",
            ],
            "action_items_heading": "बचाव के पूर्व उपाय",
            "action_items": [
                "खड़ी ढलानों, नदी नालों और घाटियों से दूर सुरक्षित ऊंचे स्थानों की ओर निकासी मार्ग पहले से तय रखें।",
                "आपातकालीन किट तैयार रखें: प्राथमिक चिकित्सा पेटी, टॉर्च, बैटरी रेडियो, सूखा खाद्य पदार्थ, पीने का पानी, जरूरी दवाएं तथा पहचान पत्र वाटरप्रूफ बैग में रखें।",
                "आपातकालीन नंबर (112, 1078, 1070) परिवार के सभी मोबाइल फोन में सुरक्षित रखें।",
                "मानसून के दौरान भारत मौसम विज्ञान विभाग (IMD) और स्थानीय प्रशासन की भारी वर्षा चेतावनियों पर निरंतर नजर रखें।",
            ],
        },
        "during": {
            "category": "भूस्खलन के दौरान (तत्काल आपातकालीन कदम)",
            "indoor_heading": "यदि आप इमारत के अंदर हैं",
            "indoor_actions": [
                "तुरंत ऊपरी मंजिल की ओर जाएं और ढलान वाली दिशा की बाहरी दीवारों से दूर रहें।",
                "किसी मजबूत फर्नीचर (जैसे भारी मेज या बेंच) के नीचे शरण लें ताकि गिरते मलबे से बचाव हो सके।",
                "खिड़कियों और कांच के दरवाजों से दूर रहें।",
            ],
            "outdoor_heading": "यदि आप खुले मैदान में हैं",
            "outdoor_actions": [
                "भूस्खलन, पत्थर गिरने या मलबे के बहाव के सीधे रास्ते से तुरंत दूर हटें।",
                "ढलान के लंबवत (दाएं या बाएं) या किसी सुरक्षित ऊंची चट्टानी चोटी की तरफ तेजी से दौड़ें।",
                "नदी घाटियों, संकरे नालों और निचले रास्तों की ओर कभी न जाएं।",
            ],
            "vehicle_heading": "यदि आप वाहन चला रहे हैं",
            "vehicle_actions": [
                "धंसी हुई सड़कों, गिरते पत्थरों और मलबे से पटे रास्तों पर विशेष सतर्कता बरतें।",
                "मलबे या पानी से घिरे पुलों और पुलियों को पार करने का प्रयास बिल्कुल न करें।",
                "वाहन को ढलान और चट्टानी दीवारों से दूर सुरक्षित खुली जगह पर खड़ा करें।",
            ],
            "trapped_heading": "यदि फंस जाएं या निकलना असंभव हो",
            "trapped_actions": [
                "अपने शरीर को गेंद की तरह मोड़ लें (भ्रूण मुद्रा) और सिर को दोनों हाथों से मजबूती से ढकें ताकि पत्थरों और मलबे से गंभीर चोट न लगे।",
            ],
        },
        "after": {
            "category": "भूस्खलन के बाद (घटना उपरांत सुरक्षा एवं राहत)",
            "safety_heading": "तत्काल सुरक्षा नियम",
            "safety_rules": [
                "भूस्खलन क्षेत्र से पूरी तरह दूर रहें। मुख्य घटना के कई घंटों या दिनों बाद भी द्वितीयक भूस्खलन और अचानक बाढ़ (फ्लैश फ्लड) का भारी खतरा बना रहता है।",
                "खतरे वाले क्षेत्र के बाहर सुरक्षित रहकर फंसे या घायल लोगों की पहचान करें; अकेले मलबे में न उतरें।",
                "राष्ट्रीय आपदा मोचन बल (NDRF), राज्य आपदा बल (SDRF) तथा स्थानीय पुलिस को घायलों के स्थान की सटीक जानकारी दें।",
                "टूटी हुई बिजली की तारों, क्षतिग्रस्त गैस पाइपों तथा पानी की लाइनों की तुरंत संबंधित विभागों को सूचना दें।",
                "जब तक सिविल इंजीनियर या जिला प्रशासन इमारत को सुरक्षित घोषित न करे, तब तक क्षतिग्रस्त मकानों में वापस न जाएं।",
                "घर में पुनः प्रवेश करने से पहले नींव, दीवारों और आसपास की ढलान पर नई दरारों की भली-भांति जांच करें।",
            ],
        },
        "emergency_contacts": [
            {"service": "राष्ट्रीय आपातकालीन हेल्पलाइन", "number": "112", "notes": "अखिल भारतीय 24x7 टोल-फ्री आपातकालीन सेवा"},
            {"service": "NDMA आपदा हेल्पलाइन", "number": "1078", "notes": "राष्ट्रीय आपदा प्रबंधन प्राधिकरण (टोल-फ्री)"},
            {"service": "राज्य आपदा प्रबंधन (SDMA)", "number": "1070", "notes": "राज्य आपातकालीन संचालन केंद्र"},
            {"service": "जिला आपदा प्रबंधन (DDMA)", "number": "1077", "notes": "जिला नियंत्रण कक्ष"},
            {"service": "पुलिस सहायता", "number": "100", "notes": "स्थानीय पुलिस नियंत्रण कक्ष"},
            {"service": "चिकित्सा एम्बुलेंस", "number": "108 / 102", "notes": "आपातकालीन चिकित्सा वाहन"},
            {"service": "अग्निशमन एवं राहत दल", "number": "101", "notes": "फायर ब्रिगेड एवं बचाव दल"},
        ],
        "disclaimer": "यह सुरक्षा मार्गदर्शिका NDMA और GSI द्वारा प्रमाणित नागरिक सुरक्षा निर्देशों पर आधारित है। किसी सक्रिय आपदा के समय हमेशा स्थानीय जिला प्रशासन, पुलिस और NDRF के निर्देशों का ही पालन करें।",
    },
    "pa": {
        "language": "pa",
        "title": "ਭੂ-ਖਿਸਕਣ (ਜ਼ਮੀਨ ਖਿਸਕਣ) ਸੁਰੱਖਿਆ ਅਤੇ ਐਮਰਜੈਂਸੀ ਮਾਰਗਦਰਸ਼ਕ",
        "authority": "ਕੌਮੀ ਆਫ਼ਤ ਪ੍ਰਬੰਧਨ ਅਥਾਰਟੀ (NDMA) ਅਤੇ ਭਾਰਤੀ ਭੂ-ਵਿਗਿਆਨਕ ਸਰਵੇਖਣ (GSI)",
        "summary": "ਪਹਾੜੀ ਇਲਾਕਿਆਂ ਵਿੱਚ ਭਾਰੀ ਮੀਂਹ ਕਾਰਨ ਹੋਣ ਵਾਲੇ ਜ਼ਮੀਨ ਖਿਸਕਣ ਅਤੇ ਮਲਬੇ ਦੇ ਵਹਾਅ ਲਈ ਪ੍ਰਮਾਣਿਤ ਸੁਰੱਖਿਆ ਨਿਯਮ।",
        "before": {
            "category": "ਭੂ-ਖਿਸਕਣ ਤੋਂ ਪਹਿਲਾਂ (ਤਿਆਰੀ ਅਤੇ ਪੂਰਵ ਚੇਤਾਵਨੀ ਸੰਕੇਤ)",
            "warning_signs_heading": "ਚੇਤਾਵਨੀ ਸੰਕੇਤਾਂ ਦੀ ਪਛਾਣ ਕਰੋ",
            "warning_signs": [
                "ਕੰਧਾਂ, ਫਰਸ਼, ਟਾਈਲਾਂ ਜਾਂ ਇਮਾਰਤ ਦੀਆਂ ਨੀਹਾਂ ਵਿੱਚ ਨਵੀਆਂ ਦਰਾਰਾਂ ਪੈਣਾ।",
                "ਦਰਵਾਜ਼ਿਆਂ ਜਾਂ ਖਿੜਕੀਆਂ ਦਾ ਅਚਾਨਕ ਜਾਮ ਹੋਣਾ ਜਾਂ ਅੜਨਾ।",
                "ਬਾਹਰੀ ਕੰਧਾਂ, ਵਾੜਾਂ, ਬਿਜਲੀ ਦੇ ਖੰਭਿਆਂ ਜਾਂ ਰੁੱਖਾਂ ਦਾ ਢਲਾਨ ਵੱਲ ਝੁਕਣਾ।",
                "ਪਹਾੜੀ ਢਲਾਨਾਂ, ਸੜਕਾਂ ਜਾਂ ਜ਼ਮੀਨ 'ਤੇ ਹੌਲੀ-ਹੌਲੀ ਚੌੜੀਆਂ ਹੁੰਦੀਆਂ ਦਰਾਰਾਂ ਦਿਖਾਈ ਦੇਣਾ।",
                "ਜ਼ਮੀਨ ਵਿੱਚੋਂ ਅਚਾਨਕ ਨਵੇਂ ਝਰਨੇ ਫੁੱਟਣਾ ਜਾਂ ਪਾਣੀ ਦਾ ਅਣਕਿਆਸੇ ਥਾਵਾਂ ਤੋਂ ਬਾਹਰ ਆਉਣਾ।",
                "ਨਦੀ-ਨਾਲਿਆਂ ਵਿੱਚ ਅਚਾਨਕ ਮਟਮੈਲਾ/ਗੰਦਾ ਪਾਣੀ ਵਹਿਣਾ ਜਾਂ ਪਾਣੀ ਦੇ ਪੱਧਰ ਵਿੱਚ ਤੇਜ਼ੀ ਨਾਲ ਤਬਦੀਲੀ ਆਉਣਾ।",
                "ਦੂਰੋਂ ਆਉਂਦੀ ਘੜਘੜਾਹਟ ਦੀ ਆਵਾਜ਼ ਦਾ ਹੌਲੀ-ਹੌਲੀ ਵਧਣਾ, ਜੋ ਮਲਬੇ ਦੇ ਖਿਸਕਣ ਦਾ ਵੱਡਾ ਸੰਕੇਤ ਹੈ।",
            ],
            "action_items_heading": "ਬਚਾਅ ਦੇ ਅਗਾਊਂ ਉਪਾਅ",
            "action_items": [
                "ਖੜ੍ਹੀਆਂ ਢਲਾਣਾਂ ਅਤੇ ਨਦੀ-ਨਾਲਿਆਂ ਤੋਂ ਦੂਰ ਸੁਰੱਖਿਅਤ ਉੱਚੀਆਂ ਥਾਵਾਂ ਵੱਲ ਨਿਕਾਸੀ ਦਾ ਰਸਤਾ ਪਹਿਲਾਂ ਹੀ ਤੈਅ ਕਰੋ।",
                "ਐਮਰਜੈਂਸੀ ਕਿੱਟ ਤਿਆਰ ਰੱਖੋ: ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਬਾਕਸ, ਟਾਰਚ, ਬੈਟਰੀ ਵਾਲਾ ਰੇਡੀਓ, ਸੁੱਕਾ ਰਾਸ਼ਨ, ਪੀਣ ਵਾਲਾ ਪਾਣੀ, ਜ਼ਰੂਰੀ ਦਵਾਈਆਂ ਅਤੇ ਸ਼ਨਾਖਤੀ ਕਾਰਡ ਵਾਟਰਪਰੂਫ ਬੈਗ ਵਿੱਚ ਰੱਖੋ।",
                "ਐਮਰਜੈਂਸੀ ਹੈਲਪਲਾਈਨ ਨੰਬਰ (112, 1078, 1070) ਪਰਿਵਾਰ ਦੇ ਸਾਰੇ ਫੋਨਾਂ ਵਿੱਚ ਸੇਵ ਰੱਖੋ।",
                "ਮਾਨਸੂਨ ਦੌਰਾਨ ਮੌਸਮ ਵਿਭਾਗ (IMD) ਅਤੇ ਜ਼ਿਲ੍ਹਾ ਪ੍ਰਸ਼ਾਸਨ ਵੱਲੋਂ ਜਾਰੀ ਭਾਰੀ ਮੀਂਹ ਦੀਆਂ ਚੇਤਾਵਨੀਆਂ 'ਤੇ ਨਜ਼ਰ ਰੱਖੋ।",
            ],
        },
        "during": {
            "category": "ਭੂ-ਖਿਸਕਣ ਦੌਰਾਨ (ਤੁਰੰਤ ਐਮਰਜੈਂਸੀ ਕਦਮ)",
            "indoor_heading": "ਜੇਕਰ ਤੁਸੀਂ ਇਮਾਰਤ ਦੇ ਅੰਦਰ ਹੋ",
            "indoor_actions": [
                "ਤੁਰੰਤ ਉੱਪਰਲੀ ਮੰਜ਼ਿਲ ਵੱਲ ਜਾਓ ਅਤੇ ਢਲਾਨ ਵਾਲੀ ਦਿਸ਼ਾ ਦੀਆਂ ਕੰਧਾਂ ਤੋਂ ਦੂਰ ਰਹੋ।",
                "ਕਿਸੇ ਮਜ਼ਬੂਤ ਫਰਨੀਚਰ (ਜਿਵੇਂ ਕਿ ਭਾਰੀ ਮੇਜ਼ ਜਾਂ ਬੈਂਚ) ਹੇਠਾਂ ਬੈਠੋ ਤਾਂ ਜੋ ਡਿੱਗਦੇ ਮਲਬੇ ਤੋਂ ਸਿਰ ਦਾ ਬਚਾਅ ਹੋ ਸਕੇ।",
                "ਖਿੜਕੀਆਂ ਅਤੇ ਸ਼ੀਸ਼ੇ ਦੇ ਦਰਵਾਜ਼ਿਆਂ ਤੋਂ ਦੂਰ ਰਹੋ।",
            ],
            "outdoor_heading": "ਜੇਕਰ ਤੁਸੀਂ ਖੁੱਲ੍ਹੇ ਮੈਦਾਨ ਵਿੱਚ ਹੋ",
            "outdoor_actions": [
                "ਜ਼ਮੀਨ ਖਿਸਕਣ, ਡਿੱਗਦੇ ਪੱਥਰਾਂ ਜਾਂ ਮਲਬੇ ਦੇ ਸਿੱਧੇ ਰਸਤੇ ਤੋਂ ਤੁਰੰਤ ਦੂਰ ਹਟੋ।",
                "ਢਲਾਨ ਦੇ ਸਮਾਨਾਂਤਰ (ਸੱਜੇ ਜਾਂ ਖੱਬੇ) ਕਿਸੇ ਉੱਚੀ ਪੱਕੀ ਚੱਟਾਨ ਵੱਲ ਤੇਜ਼ੀ ਨਾਲ ਦੌੜੋ।",
                "ਨਦੀ ਘਾਟੀਆਂ, ਤੰਗ ਖੱਡਾਂ ਅਤੇ ਨੀਵੇਂ ਨਾਲਿਆਂ ਵੱਲ ਕਦੇ ਨਾ ਜਾਓ।",
            ],
            "vehicle_heading": "ਜੇਕਰ ਤੁਸੀਂ ਗੱਡੀ ਚਲਾ ਰਹੇ ਹੋ",
            "vehicle_actions": [
                "ਧਸੀਆਂ ਸੜਕਾਂ, ਡਿੱਗਦੇ ਪੱਥਰਾਂ ਅਤੇ ਮਲਬੇ ਵਾਲੀਆਂ ਸੜਕਾਂ 'ਤੇ ਖਾਸ ਨਜ਼ਰ ਰੱਖੋ।",
                "ਮਲਬੇ ਜਾਂ ਪਾਣੀ ਹੇਠ ਆਏ ਪੁਲਾਂ ਅਤੇ ਪੁਲੀਆਂ ਨੂੰ ਪਾਰ ਕਰਨ ਦੀ ਕੋਸ਼ਿਸ਼ ਨਾ ਕਰੋ।",
                "ਗੱਡੀ ਨੂੰ ਢਲਾਨਾਂ ਅਤੇ ਕਮਜ਼ੋਰ ਚੱਟਾਨਾਂ ਤੋਂ ਦੂਰ ਖੁੱਲ੍ਹੀ ਅਤੇ ਪੱਕੀ ਥਾਂ 'ਤੇ ਰੋਕੋ।",
            ],
            "trapped_heading": "ਜੇਕਰ ਫਸ ਜਾਓ ਜਾਂ ਨਿਕਲਣਾ ਅਸੰਭਵ ਹੋਵੇ",
            "trapped_actions": [
                "ਆਪਣੇ ਸਰੀਰ ਨੂੰ ਗੇਂਦ ਵਾਂਗ ਇਕੱਠਾ ਕਰ ਲਵੋ (ਭਰੂਣ ਸਥਿਤੀ) ਅਤੇ ਦੋਵਾਂ ਹੱਥਾਂ ਨਾਲ ਆਪਣੇ ਸਿਰ ਨੂੰ ਮਜ਼ਬੂਤੀ ਨਾਲ ਢਕੋ ਤਾਂ ਜੋ ਪੱਥਰਾਂ ਦੀ ਚੋਟ ਤੋਂ ਬਚਿਆ ਜਾ ਸਕੇ।",
            ],
        },
        "after": {
            "category": "ਭੂ-ਖਿਸਕਣ ਤੋਂ ਬਾਅਦ (ਹਾਦਸੇ ਉਪਰੰਤ ਸੁਰੱਖਿਆ ਅਤੇ ਰਾਹਤ)",
            "safety_heading": "ਤੁਰੰਤ ਸੁਰੱਖਿਆ ਨਿਯਮ",
            "safety_rules": [
                "ਭੂ-ਖਿਸਕਣ ਵਾਲੇ ਖੇਤਰ ਤੋਂ ਪੂਰੀ ਤਰ੍ਹਾਂ ਦੂਰ ਰਹੋ। ਮੁੱਖ ਘਟਨਾ ਤੋਂ ਕਈ ਘੰਟੇ ਜਾਂ ਦਿਨਾਂ ਬਾਅਦ ਵੀ ਦੁਬਾਰਾ ਜ਼ਮੀਨ ਖਿਸਕਣ ਅਤੇ ਅਚਾਨਕ ਹੜ੍ਹ ਆਉਣ ਦਾ ਗੰਭੀਰ ਖ਼ਤਰਾ ਰਹਿੰਦਾ ਹੈ।",
                "ਖ਼ਤਰੇ ਵਾਲੇ ਜ਼ੋਨ ਤੋਂ ਬਾਹਰ ਰਹਿ ਕੇ ਫਸੇ ਜਾਂ ਜ਼ਖਮੀ ਲੋਕਾਂ ਦੀ ਪਛਾਣ ਕਰੋ; ਇਕੱਲੇ ਮਲਬੇ ਵਿੱਚ ਨਾ ਵੜੋ।",
                "ਰਾਸ਼ਟਰੀ ਆਫ਼ਤ ਰਿਸਪਾਂਸ ਫੋਰਸ (NDRF), ਰਾਜ ਆਫ਼ਤ ਫੋਰਸ (SDRF) ਅਤੇ ਸਥਾਨਕ ਪੁਲਿਸ ਨੂੰ ਜ਼ਖਮੀਆਂ ਦੀ ਸਹੀ ਜਾਣਕਾਰੀ ਦਿਓ।",
                "ਟੁੱਟੀਆਂ ਬਿਜਲੀ ਦੀਆਂ ਤਾਰਾਂ, ਖ਼ਰਾਬ ਗੈਸ ਪਾਈਪਾਂ ਅਤੇ ਪਾਣੀ ਦੀਆਂ ਪਾਈਪਾਂ ਬਾਰੇ ਤੁਰੰਤ ਅਧਿਕਾਰੀਆਂ ਨੂੰ ਸੂਚਿਤ ਕਰੋ।",
                "ਜਦੋਂ ਤੱਕ ਇੰਜੀਨੀਅਰ ਜਾਂ ਜ਼ਿਲ੍ਹਾ ਪ੍ਰਸ਼ਾਸਨ ਇਮਾਰਤ ਨੂੰ ਸੁਰੱਖਿਅਤ ਨਾ ਕਹੇ, ਨੁਕਸਾਨੇ ਗਏ ਘਰਾਂ ਵਿੱਚ ਵਾਪਸ ਨਾ ਜਾਓ।",
                "ਘਰ ਵਿੱਚ ਮੁੜ ਦਾਖਲ ਹੋਣ ਤੋਂ ਪਹਿਲਾਂ ਨੀਹਾਂ, ਕੰਧਾਂ ਅਤੇ ਆਲੇ-ਦੁਆਲੇ ਦੀ ਜ਼ਮੀਨ ਦੀ ਚੰਗੀ ਤਰ੍ਹਾਂ ਜਾਂਚ ਕਰੋ।",
            ],
        },
        "emergency_contacts": [
            {"service": "ਕੌਮੀ ਐਮਰਜੈਂਸੀ ਹੈਲਪਲਾਈਨ", "number": "112", "notes": "ਸਾਰੇ ਭਾਰਤ ਲਈ 24x7 ਟੋਲ-ਫ੍ਰੀ ਐਮਰਜੈਂਸੀ ਸੇਵਾ"},
            {"service": "NDMA ਆਫ਼ਤ ਹੈਲਪਲਾਈਨ", "number": "1078", "notes": "ਕੌਮੀ ਆਫ਼ਤ ਪ੍ਰਬੰਧਨ ਅਥਾਰਟੀ (ਟੋਲ-ਫ੍ਰੀ)"},
            {"service": "ਸੂਬਾ ਆਫ਼ਤ ਪ੍ਰਬੰਧਨ (SDMA)", "number": "1070", "notes": "ਸੂਬਾਈ ਐਮਰਜੈਂਸੀ ਕੰਟਰੋਲ ਰੂਮ"},
            {"service": "ਜ਼ਿਲ੍ਹਾ ਆਫ਼ਤ ਪ੍ਰਬੰਧਨ (DDMA)", "number": "1077", "notes": "ਜ਼ਿਲ੍ਹਾ ਕੰਟਰੋਲ ਰੂਮ"},
            {"service": "ਪੁਲਿਸ ਸਹਾਇਤਾ", "number": "100", "notes": "ਸਥਾਨਕ ਪੁਲਿਸ ਕੰਟਰੋਲ ਰੂਮ"},
            {"service": "ਐਂਬੂਲੈਂਸ ਸੇਵਾ", "number": "108 / 102", "notes": "ਮੈਡੀਕਲ ਐਮਰਜੈਂਸੀ ਵਾਹਨ"},
            {"service": "ਫਾਇਰ ਬ੍ਰਿਗੇਡ ਅਤੇ ਰਾਹਤ ਦਲ", "number": "101", "notes": "ਅੱਗ ਬੁਝਾਊ ਅਤੇ ਬਚਾਅ ਟੀਮ"},
        ],
        "disclaimer": "ਇਹ ਸੁਰੱਖਿਆ ਨਿਯਮ NDMA ਅਤੇ GSI ਵੱਲੋਂ ਪ੍ਰਮਾਣਿਤ ਨਾਗਰਿਕ ਸੁਰੱਖਿਆ ਦਿਸ਼ਾ-ਨਿਰਦੇਸ਼ਾਂ 'ਤੇ ਆਧਾਰਿਤ ਹਨ। ਕਿਸੇ ਸਰਗਰਮ ਆਫ਼ਤ ਵੇਲੇ ਹਮੇਸ਼ਾ ਜ਼ਿਲ੍ਹਾ ਪ੍ਰਸ਼ਾਸਨ, ਪੁਲਿਸ ਅਤੇ NDRF ਦੇ ਨਿਰਦੇਸ਼ਾਂ ਦੀ ਪਾਲਣਾ ਕਰੋ।",
    },
}


def normalize_language_code(lang: str | None) -> str:
    """Normalize language name or code to a supported 2-letter key ('en', 'hi', 'pa')."""
    if not lang:
        return "en"
    clean = lang.strip().lower()
    if clean in ("en", "english"):
        return "en"
    if clean in ("hi", "hindi", "हिंदी", "हिन्दी"):
        return "hi"
    if clean in ("pa", "punjabi", "panjabi", "ਪੰਜਾਬੀ"):
        return "pa"
    return "en"


def get_safety_guidance(language: str = "en") -> dict[str, Any]:
    """Retrieve verified landslide safety guidance for the specified language.

    Args:
        language: Language code or name ('en', 'hi', 'pa'). Defaults to 'en'.

    Returns:
        Structured safety guidance dictionary with verified civil-defense advice.
    """
    code = normalize_language_code(language)
    return SAFETY_DATABASE.get(code, SAFETY_DATABASE["en"])


def get_supported_languages() -> list[dict[str, str]]:
    """Return list of supported language descriptors."""
    return list(SUPPORTED_LANGUAGES.values())


def format_safety_markdown(guidance: dict[str, Any]) -> str:
    """Format structured safety guidance dictionary into clean GitHub Flavored Markdown."""
    lines: list[str] = []
    lines.append(f"## 🛡️ {guidance['title']}")
    lines.append(f"> **Authority:** {guidance['authority']}  ")
    lines.append(f"> {guidance['summary']}\n")

    # Before
    b = guidance["before"]
    lines.append(f"### 🟡 {b['category']}")
    lines.append(f"**{b['warning_signs_heading']}**")
    for ws in b["warning_signs"]:
        lines.append(f"- ⚠️ {ws}")
    lines.append(f"\n**{b['action_items_heading']}**")
    for ai in b["action_items"]:
        lines.append(f"- ✅ {ai}")
    lines.append("")

    # During
    d = guidance["during"]
    lines.append(f"### 🔴 {d['category']}")
    lines.append(f"**{d['indoor_heading']}**")
    for ia in d["indoor_actions"]:
        lines.append(f"- 🏠 {ia}")
    lines.append(f"\n**{d['outdoor_heading']}**")
    for oa in d["outdoor_actions"]:
        lines.append(f"- 🏃 {oa}")
    lines.append(f"\n**{d['vehicle_heading']}**")
    for va in d["vehicle_actions"]:
        lines.append(f"- 🚗 {va}")
    lines.append(f"\n**{d['trapped_heading']}**")
    for ta in d["trapped_actions"]:
        lines.append(f"- 🛡️ {ta}")
    lines.append("")

    # After
    a = guidance["after"]
    lines.append(f"### 🟢 {a['category']}")
    lines.append(f"**{a['safety_heading']}**")
    for sr in a["safety_rules"]:
        lines.append(f"- 🔍 {sr}")
    lines.append("")

    # Emergency Contacts
    lines.append("### 📞 Emergency Helplines (India)")
    lines.append("| Service / Agency | Number | Details |")
    lines.append("|---|---|---|")
    for ec in guidance["emergency_contacts"]:
        lines.append(f"| **{ec['service']}** | `{ec['number']}` | {ec['notes']} |")
    lines.append("")

    # Disclaimer
    lines.append(f"> [!CAUTION]\n> {guidance['disclaimer']}\n")
    return "\n".join(lines)

