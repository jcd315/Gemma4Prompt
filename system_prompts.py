"""
System prompts, presets, and message-building utilities for Gemma4PromptGen.

Contains:
  - ENVIRONMENT_PRESETS: ~70 environment tuples (location, lighting, sound)
  - ANIMATION_PRESETS: 24 cartoon/animation universe definitions
  - TARGET_MODELS: model dropdown options
  - System prompt strings for each target model (LTX, Wan, Flux, SDXL, Pony, SD1.5)
  - get_system_prompt(), is_video_model(), has_audio() helpers
  - build_user_message() — assembles the full user prompt message
  - clean_llm_output() — cleans raw LLM output into usable prompts
"""

import re
import random


# ══════���═════════════════════════════════════════���═════════════════════════
#  ENVIRONMENT PRESETS
#  Each value: (location, lighting, sound)
#  None = LLM decides.  "RANDOM" = seed-picked at runtime.
# ══════════════════���═══════════════════════════════════════════════════════
ENVIRONMENT_PRESETS = {
    "None — LLM decides": None,
    "🎲 Random — seed picks": "RANDOM",

    # ── NATURAL ───────���──────────────────────────────────────────────────
    "🏖 Beach — golden hour": (
        "wide open beach at golden hour, warm amber light raking low across wet sand, "
        "shallow surf foaming in irregular sheets over the flat shore, "
        "distant horizon blurred with sea haze, seaweed and shell fragments at the tide line, "
        "salt crust on every exposed surface, damp sand firm underfoot then soft further up the beach",
        "warm directional sidelight from the low sun, long soft shadows stretching inland, "
        "orange-gold palette with deep blue shadows pooling in the wet sand troughs",
        "rolling waves building and collapsing, wind-carried spray hissing across the sand, "
        "distant gulls, the hollow clap of a wave folding on itself"),

    "🏔 Mountain peak — dawn": (
        "exposed mountain summit at first light, vast sky opening below in every direction, "
        "cold thin air, bare grey-brown rock underfoot fractured into angular plates, "
        "pale blue and rose light spreading from the east across cloud layers far below, "
        "distant ranges stretching to a gently curved horizon, breath visible in the cold",
        "cold directional dawn light from the east, high contrast, no fill light, "
        "long purple shadows from every ridge and rock formation, rose-to-blue sky gradient",
        "wind building and fading in slow gusts, deep silence between them, "
        "the creak of cold rock contracting, faint echo from the valley below"),

    "🌲 Dense forest — diffused green": (
        "deep forest interior, canopy dense and fully closed 20 metres overhead, "
        "light filtering down in soft broken columns through layered leaves, "
        "moss-covered ground, ferns at knee height filling every gap between roots, "
        "standing water in root depressions reflecting green light back upward, "
        "bark textured with lichen and fungal rings, the space between trunks creating receding depth",
        "diffused green-filtered light with no hard shadows, uniform soft fill from the canopy above, "
        "every surface tinted with reflected chlorophyll green",
        "birdsong in overlapping species layers, wind audible in the canopy but absent at ground level, "
        "a dry leaf shifting somewhere unseen, distant running water"),

    "🌊 Underwater — shallow reef": (
        "shallow tropical reef underwater, clear turquoise water with 20-metre visibility, "
        "shafts of broken sunlight refracting through the rippling surface in caustic patterns, "
        "staghorn and brain coral formations in soft focus below, "
        "small fish holding station in the gentle current, everything moving in slow surge rhythm",
        "caustic light patterns dancing across every surface from above, "
        "high-key teal-blue overall, darker blue fading into depth below",
        "muffled pressure, the steady rise of bubbles, distant boat hull drone, "
        "the creak of coral in the current"),

    "🌧 Rain-soaked city street — night": (
        "rain-soaked urban street at night, wet asphalt reflecting neon signs "
        "in elongated distorted colour streaks, steam rising from iron grates in the road, "
        "pools of amber streetlight surrounded by dark, blurred traffic in background, "
        "awnings dripping, gutters running",
        "neon colour reflections in puddles — red, blue, white, amber — "
        "cool blue ambient fill, warm sodium overhead streetlamps",
        "rain on pavement in constant hiss, distant traffic, "
        "wet tyre sound on asphalt, footsteps echoing under an awning"),

    "🏜 Desert — midday heat": (
        "open desert at midday, bleached pale sand extending to a dead-flat horizon, "
        "air rippling with heat shimmer low above the ground, "
        "sky a brilliant white-blue with no cloud, no shade, no landmarks, "
        "surface cracked into geometric plates closer to the foreground",
        "brutal overhead sun, harsh vertical top-light with zero shadow relief, "
        "bleached palette — near-white sand, white-blue sky, black under anything that casts shade",
        "silence — then wind — then silence again, fine sand skittering across the crust"),

    "🌌 Night sky — open field": (
        "open field under a fully clear night sky, grass running to a dark horizon, "
        "the Milky Way arcing overhead in a dense band of blue-white stars, "
        "no artificial light source, ground-level detail barely visible in deep blue-black ambient",
        "starlight only, near-black ambient, faint blue-grey top-light from the sky itself, "
        "the Milky Way core casting a measurable soft gradient",
        "crickets in continuous layers, light wind through the grass, "
        "a frog somewhere, the profound silence beneath everything"),

    "🌁 Rooftop — city at night": (
        "high rooftop at night, city skyline spreading in every direction below, "
        "warm glow rising from the streets like a second horizon, "
        "wind at this height, ventilation stacks and water tanks breaking the flat roof surface, "
        "a parapet at the edge with the drop visible beyond it",
        "city glow from below as warm amber fill, cool blue sky above, "
        "backlit silhouette potential against the lit skyline",
        "distant city hum rising and falling, wind, "
        "an occasional siren rising from far below and fading"),

    "✈ Plane cockpit — cruising altitude": (
        "aircraft cockpit at cruising altitude, instrument panel spread in amber and green glow, "
        "black sky through the windshield, stars visible above the cloud layer, "
        "the vibration and low hum of engines constant beneath everything, "
        "oxygen mask clips and circuit breakers detailed on the overhead panel",
        "instrument panel glow from below — warm amber dials, green digital readouts — "
        "cool black from the windshield, no natural light",
        "engine hum constant and enveloping, radio static between calls, "
        "pressurised air hiss from the vents, the occasional click of switches"),

    # ── INTERIOR ────────────────────────────────��────────────────────────
    "�� Bedroom — warm evening": (
        "warm bedroom interior in the evening, a single bedside lamp casting a pool of amber light, "
        "soft shadow in the far corners, bed linen slightly rumpled with the weight of use, "
        "curtains drawn against the dark outside, a glass of water on the nightstand",
        "warm tungsten point source from the bedside lamp, soft falloff, "
        "intimate amber glow, deep shadow beyond its reach",
        "rain against the window glass if it's raining, or the distant low hum of the city through double glazing, "
        "the bed shifting under weight, fabric sliding on fabric, "
        "a phone on the nightstand screen briefly lighting then going dark, "
        "breathing — the rhythm and depth of it — the only sound that belongs to the room itself"),

    "🛁 Bathroom — steam and tile": (
        "steam-filled bathroom, a hot shower running behind frosted glass, "
        "white tile walls beaded with condensation, mirror completely fogged over, "
        "damp warm air thick enough to see, a folded towel on the rail, "
        "soap residue on the tile floor",
        "diffused warm light through frosted glass — soft, hazy, no hard edges, "
        "the steam itself lit from within",
        "shower hiss steady behind glass, water hitting tile, "
        "a slow drip from the tap, muffled echo in the tiled space"),

    "🪟 Penthouse — floor-to-ceiling glass": (
        "high-floor penthouse interior with floor-to-ceiling glass on two walls, "
        "city spread far below, clean minimal interior — low furniture in dark leather and pale stone, "
        "daylight flooding in from the glass wall, the room reflected in the glass at certain angles",
        "natural daylight through glass — even, cool, diffused by height and haze — "
        "city providing a continuous ambient glow from below at night",
        "near-silence — the city thirty floors below reduced to a formless low frequency hum, "
        "the building's HVAC cycling barely audible, glass creaking faintly in wind at this height, "
        "ice settling in a glass, the sound of someone's breathing amplified by the quiet, "
        "and the occasional deep resonant vibration of the building itself moving"),

    "🎹 Jazz club — late night": (
        "intimate jazz club late at night, low ceiling with exposed brickwork, "
        "small stage lit warm at the far end, tables pressed close together, "
        "a candle stub on each table burning low, smoke visible in the stage light, "
        "a bar along one wall with backlit bottles",
        "warm tungsten stage wash, candle fill table by table, "
        "deep shadow in the corners and upper walls",
        "a jazz trio — upright bass, brushed snare, and a tenor saxophone — playing a slow blues "
        "at the far end of the room, the saxophone filling the space and bending at the end of each phrase, "
        "the bassist walking the changes in a low steady pulse, brushes on the snare barely louder than breathing, "
        "a glass set down on the bar between phrases, low conversation that stops "
        "when the sax player leans into a long held note, "
        "the specific intimate acoustic of a low ceiling that puts the music right inside the chest"),

    "🚂 Train — moving through night": (
        "train carriage moving at night, window showing dark landscape "
        "with scattered lights passing in rhythm, warm interior against the cold black outside, "
        "moving reflections of the carriage interior in the glass, "
        "seats in worn fabric, the rhythmic sway of the carriage",
        "warm interior tungsten against total black window exterior, "
        "moving reflections layered over the dark passing world",
        "rhythmic track click accelerating and decelerating on curves, "
        "engine vibration through the floor, the world passing outside muffled by glass"),

    "💊 Underground club — strobes and bass": (
        "underground club at full capacity, strobes cutting the dark in sharp white intervals, "
        "bass pressure felt in the chest before it is heard, crowd pressed together in the dark, "
        "a DJ booth visible through smoke at the far end, coloured wash lights sweeping low",
        "stroboscopic white cuts, colour wash through smoke — purple, red, blue — "
        "near-black between flashes, faces caught in freeze-frame light",
        "bass at physical volume, the crowd as a breathing mass of sound, "
        "the specific compression of a room built for this volume"),

    "🏢 Office — after hours": (
        "corporate office after hours, desks empty and personal items abandoned mid-day, "
        "flat cold overhead fluorescent across an open-plan floor, "
        "city visible through floor-to-ceiling glass on one wall, "
        "the quality of silence that fills a building after everyone has left",
        "flat cold fluorescent overhead, warm city glow through the glass, "
        "clinical blue-white palette, long shadows from desk furniture",
        "air conditioning hum at low frequency, a distant elevator, "
        "the silence of an empty building with one person in it"),

    "🚗 Car — moving at night": (
        "car interior at night, moving through a lit city, streetlights sweeping "
        "through the windows in rhythmic pulses of amber and shadow, "
        "dashboard instruments glowing warm from below, city blurred and wet outside, "
        "the close interior smell of upholstery and warm electronics",
        "rhythmic streetlight sweeps through the windows, "
        "warm dashboard glow from below, moving pattern of light and shadow across interior surfaces",
        "engine, tyres on wet road, city muffled by glass, "
        "faint radio under everything"),

    # ── ICONIC LOCATIONS ─────────────────────────────────────────────────
    "🏰 Big Ben — Westminster at night": (
        "standing directly beneath the Elizabeth Tower on the Westminster Bridge approach, "
        "the illuminated clock face filling the upper frame, warm floodlit limestone glowing gold "
        "against a deep navy sky, the Thames visible beyond the stone parapet, "
        "black iron lampposts lining the bridge behind, black cabs and buses passing in soft blur",
        "warm sodium floodlighting on the tower face, cold blue ambient sky, "
        "wet stone reflecting gold below, the clock face its own light source",
        "distant Big Ben chime on the quarter, Thames wind across the bridge, "
        "traffic crossing behind, footsteps on stone"),

    "🗽 Times Square — peak night": (
        "standing at the centre of Times Square at 2am, surrounded by skyscrapers "
        "sheathed in animated LED billboards — saturated reds, whites, yellows cascading down the canyon walls, "
        "NASDAQ ticker scrolling, yellow cabs streaming through the intersection, "
        "tourists in every direction, steam rising from road grates",
        "total ambient saturation — no single source, light arriving simultaneously from every direction, "
        "colour-shifting as the billboards cycle",
        "traffic, crowd hum, distant busker, NYPD siren one block over, "
        "the specific sound of a city that never quietens"),

    "🗼 Eiffel Tower — sparkling midnight": (
        "standing on the Champ de Mars facing the Eiffel Tower at midnight, "
        "the hourly light show in full effect — 20,000 gold bulbs sparkling in random sequence "
        "across the iron lattice, the Seine to the left catching the glow, "
        "Parisian apartment blocks framing both sides, a few couples on the lawn nearby",
        "gold sparkle wash from the tower varying every second, "
        "deep blue ambient sky, distant street lamp orange at the park edges",
        "city ambience, wind across the park, the faint metallic creak of the iron structure, "
        "distant traffic on the quai"),

    "🌉 Golden Gate Bridge — fog morning": (
        "standing mid-span on the Golden Gate Bridge walkway, "
        "thick morning fog rolling in from the Pacific and swallowing the south tower completely, "
        "only the top third of the north tower visible above the fog line, "
        "the bridge roadway disappearing into white in both directions, "
        "the bay invisible below, cold salt air, the bridge's suspension cables vanishing into cloud",
        "flat diffuse fog light — directionless, grey-white, no shadows, "
        "every surface equally softened, the towers fading to silhouette then to nothing",
        "wind through the cables producing a low resonant hum that changes pitch with gusts, "
        "foghorn in the bay, distant muffled traffic"),

    "🏯 Japanese shrine — early morning": (
        "ancient Shinto shrine at first light, stone torii gate at the entrance "
        "casting a long shadow down the gravel path, stone lanterns lining both sides, "
        "cedar trees so tall the canopy closes overhead, "
        "moss on every stone surface, a single paper lantern still lit from overnight at the main hall",
        "cool blue pre-dawn light filtering through cedar, "
        "warm paper lantern glow at the gate, raking first light beginning on the gravel",
        "wind through cedar boughs, gravel shifting underfoot, "
        "distant temple bell, water dripping from a stone basin"),

    "🌆 Tokyo Shibuya crossing — night": (
        "the Shibuya scramble crossing at night between signal changes, "
        "hundreds of people streaming in every direction simultaneously, "
        "Shibuya 109 building and its neon crown directly ahead, "
        "rain-slicked asphalt reflecting every sign and screen in doubled colour, "
        "7-Eleven and Starbucks glowing warm through steam",
        "neon and LED saturation from every angle — amber, white, red, blue — no hard shadows, "
        "everything doubled in the wet ground",
        "crossing signal tone, crowd footsteps, idling cars, "
        "distant J-pop from a store entrance, the specific density of Shibuya at night"),

    "🌊 Amalfi Coast — cliff road": (
        "narrow coastal road cut directly into the Amalfi cliff face, "
        "turquoise Mediterranean far below catching direct sun and breaking white on the rocks, "
        "no barrier on the seaward side of the road, "
        "lemon groves terraced into the hillside above, "
        "a white-painted village visible across the bay in the afternoon haze",
        "Mediterranean full sun — hard, directional, high contrast, "
        "deep shadows in the cliff cuts, warm gold on the road surface",
        "sea wind, waves far below, a distant scooter engine, "
        "cicadas in the lemon trees above"),

    "🏖 Maldives — overwater bungalow at dusk": (
        "wooden deck extending directly over the lagoon from an overwater bungalow, "
        "water below so clear the sand and coral are visible in turquoise and white, "
        "dusk turning the horizon to a band of orange fading through pink to violet, "
        "the Indian Ocean completely flat, other bungalows in a line behind, "
        "a rope ladder descending from the deck edge into the glowing water",
        "last light warm orange from the horizon, cool violet sky above, "
        "water reflecting both colours simultaneously",
        "water lapping at the stilts below the deck in slow irregular rhythm, "
        "a wind chime on the bungalow moving in the sea breeze, "
        "a distant boat engine somewhere out on the lagoon, "
        "the reef making its evening clicks and pops beneath the surface, "
        "a fruit bat passing overhead, and underneath all of it the oceanic silence of open water at dusk"),

    "🎪 Coachella — main stage sunset": (
        "main Coachella stage at golden hour, the Indio desert stretching to the horizon behind the crowd, "
        "mountains blue and distant in the haze, the stage framed by its giant LED screen "
        "showing warm amber graphics matching the sunset, "
        "tens of thousands on the flat desert floor, dust haze in the air, flags and totems swaying",
        "golden hour desert sun from the west, warm amber fill from the stage screens, "
        "everything amber-soaked and backlit",
        "festival crowd roar, bass from the PA crossing the desert, "
        "the dry desert wind, helicopter overhead"),

    "🌃 Seoul Han River bridge — night": (
        "walking the pedestrian lane of the Banpo Bridge at night, "
        "Seoul's skyline reflected in the Han River below in a long shimmering stripe, "
        "the Moonlight Rainbow Fountain arcing jets of lit water from the bridge rail, "
        "apartment towers in every direction, Namsan Tower with its crown visible on the hill",
        "bridge lighting warm white, fountain colour wash cycling, "
        "Seoul skyline ambient glow on the water surface",
        "water jets from the fountain, Han River wind, "
        "distant city, a passing tour boat"),

    "🏔 High-altitude snowfield": (
        "open snowfield at high altitude, no trees, no shelter, "
        "snow surface wind-sculpted into slow sastrugi waves, "
        "a single ridge of darker rock breaking the white in the far distance, "
        "sky a deep near-violet blue at this altitude, "
        "breath visible in long plumes, footstep tracks the only mark on the surface",
        "flat overcast bounce off the snow — sourceless, directionless white light, "
        "everything equally lit, no shadows, the snow itself the only light source",
        "wind — and nothing else — occasionally a snow grain skittering across the surface crust"),

    "🚇 NYC subway platform — 3am": (
        "empty New York City subway platform at 3am, "
        "tiled walls in grimy institutional cream and brown, "
        "fluorescent tubes overhead with one flickering on a slow cycle, "
        "gum-stained concrete, yellow warning stripe at the platform edge, "
        "a distant rumble building to a full roar as a train approaches and passes without stopping",
        "flat fluorescent overhead, one tube flickering, "
        "the train's headlight briefly sweeping the tunnel end",
        "train rumble building and fading, platform PA echo, "
        "a distant busker's note floating from the next platform"),

    "🌅 Santorini caldera — dawn": (
        "whitewashed terrace on the caldera rim in Santorini at first light, "
        "the volcanic caldera dropping sheer below, the Aegean spread to the horizon in deep blue, "
        "blue-domed churches clustered on the clifftop in the middle distance, "
        "bougainvillea cascading over the terrace wall in magenta",
        "first light pale gold on the white walls, deep blue sea and sky, "
        "magenta flower accent, the white walls almost glowing",
        "Aegean wind, a distant church bell, a boat engine somewhere far below"),

    "🏟 Empty stadium — floodlit night": (
        "standing alone on the pitch of a major football stadium at night, no crowd, "
        "the four giant floodlight rigs pouring hard white light down onto the turf, "
        "stands empty in darkness beyond the light line, "
        "the pitch surface wet from sprinklers, scoreboard dark",
        "four-point overhead flood — hard white industrial light, "
        "deep shadow in the empty stands beyond the light boundary",
        "floodlight hum at low constant frequency, wind across the open bowl, "
        "a single flag snapping on the roof"),

    "🎻 Vienna opera house — empty stage": (
        "standing alone on the stage of the Vienna State Opera between performances, "
        "grand proscenium arch overhead, six tiers of red velvet boxes receding into darkness, "
        "a single work light — a bare bulb on a stand — the only source on stage, "
        "the ghost light casting long shadows across the boards",
        "single bare bulb ghost light — hard, warm, tungsten — "
        "everything else in dense theatrical dark, the boxes invisible",
        "the ghost light's single bulb humming faintly at low frequency, "
        "the vast room holding its breath — the acoustic of 2000 empty velvet seats absorbing all reflection, "
        "a board creaking once under shifting weight, "
        "the heating system deep in the walls ticking, "
        "and the profound specific silence of a concert hall built for music "
        "when the music has stopped — a silence with shape and texture"),

    "🌿 Amazon jungle interior": (
        "deep Amazon rainforest interior with no sky visible, "
        "canopy 40 metres overhead and fully closed, "
        "light arriving only as occasional single shafts breaking through the layers, "
        "forest floor a tangle of buttress roots and fern, "
        "something moving in the mid-canopy unseen and continuous",
        "green-filtered indirect light, permanent green shade, "
        "occasional single shaft of direct sun breaking through, "
        "everything in the same flat green ambient",
        "constant insect layer at full volume — the Amazon roar — "
        "bird calls cutting through, distant water, drip from leaves"),

    "🧊 Ice hotel — Lapland": (
        "interior of an ice hotel room in Lapland in deep winter, "
        "walls, ceiling and furniture carved entirely from glacier ice, "
        "sleeping reindeer skins draped over ice bed frames, "
        "the walls faintly glowing blue-white from ice thickness, "
        "breath visible in every shot, everything translucent",
        "ambient blue-white glow through the ice walls — sourceless, cold, crystalline — "
        "no artificial light, the ice itself luminous",
        "near-total silence — only the creak of settling ice and breath, "
        "occasionally the distant howl of wind outside"),

    "🏬 Tokyo convenience store — 3am": (
        "Lawson or 7-Eleven interior in Tokyo at 3am, completely deserted, "
        "fluorescent lights at full brightness, every shelf perfectly faced and stocked, "
        "hot foods rotating in their case by the register, "
        "rain audible on the pavement outside, "
        "the automatic door briefly opening to let in cold air and admit no one",
        "flat harsh fluorescent overhead — clinical white, no shadows, "
        "everything overlit in that specific convenience store way",
        "refrigerator hum, hot case motor, rain outside, "
        "the door's pneumatic hiss and seal"),

    "🛕 Angkor Wat — golden hour": (
        "standing at the western causeway of Angkor Wat at sunrise, "
        "the five towers reflected in the rectangular moat below, "
        "warm orange light catching every carved sandstone spire, "
        "jungle visible above the outer walls in every direction, "
        "lotus blossoms floating on the moat surface, a monk crossing in the distance",
        "direct low sunrise orange from the east, long shadows down the causeway, "
        "warm pink sky reflected in still water",
        "jungle birds, water lapping the moat edge, distant chanting, "
        "the complete stillness of early morning before tourists arrive"),

    # ── HORROR ───────────────────────────────────────────────────────────
    "🏚 Abandoned building — dark interior": (
        "derelict interior — a former house or institution stripped back to bare structure, "
        "plaster fallen from walls exposing dark brick, floorboards rotted through in patches, "
        "a single doorway open to a deeper corridor beyond, debris underfoot, "
        "curtains torn and hanging at a broken window, rust stains tracking down every wall",
        "single motivated light source only — a torch beam, a crack of moonlight through a board, "
        "a bare bulb on a frayed wire just barely working — everything beyond its reach is near-black",
        "structural settling sounds — a distant creak, something dripping, wind through a gap, "
        "the specific silence of a space that hasn't had a person in it for years — then it does"),

    "🏥 Hospital corridor — fluorescent night": (
        "long hospital corridor at night, linoleum floor with a worn central track, "
        "institutional cream walls with a dado rail at waist height, "
        "a row of numbered doors receding in both directions, one door ajar at the far end, "
        "an overturned wheelchair near the nurse's station, a clipboard on the floor",
        "overhead fluorescent strip lights — two out, one flickering at irregular intervals, "
        "the working ones casting cold blue-white, long green-tinged shadows on the floor",
        "the flicker hum of the failing fluorescent, distant HVAC, a door somewhere closing softly, "
        "the squeak of something on the linoleum floor at the far end of the corridor"),

    "🌲 Haunted woods — dead of night": (
        "dense forest at night, canopy completely blocking the sky, "
        "bare or near-bare trees with high branches interlocked overhead, "
        "root-broken ground underfoot, a faint path barely distinguishable from the surrounding forest floor, "
        "mist at knee height in a clearing visible through the trees ahead, "
        "a structure — the suggestion of one — barely visible in the dark beyond the clearing",
        "no ambient light — torch beam only, or moonlight arriving at odd angles through gaps in the canopy, "
        "blue-black shadow everywhere the light doesn't reach, mist catching and holding any beam",
        "wind in the upper canopy — audible but not felt at ground level — "
        "an owl somewhere, a branch snapping under weight in a direction the camera hasn't looked yet"),

    # ── SPAGHETTI WESTERN ──────���─────────────────────────────────────────
    "🏜 Ghost town — high noon standoff": (
        "abandoned western town, main street wide enough for a wagon, "
        "false-front wooden buildings on both sides — general store, saloon, sheriff's office — "
        "all long-abandoned, paint peeling, a tumbleweed lodged against a hitching post, "
        "dust rising from the street in a slow gust, shutters banging on a broken hinge, "
        "a single figure at each end of the street, heat shimmer between them",
        "brutal noon sun directly overhead — no shadow, no relief, every surface bleached near-white, "
        "sky a deep saturated blue with no cloud, the sun itself the only light source",
        "wind — nothing else — then the wind stops — then silence so deep the heartbeat is audible — "
        "a shutter bangs once — and then nothing again"),

    "🌵 Open desert — late afternoon heat": (
        "flat desert extending to a dead horizon in every direction, "
        "cracked salt flats closer, red dust further out, a distant mesa barely distinguishable from sky, "
        "a single dead tree on the left edge of frame, a buzzard circling high, "
        "heat shimmer turning the horizon liquid, no road, no structure, no shade anywhere",
        "late afternoon sun at 20° — long amber shadows stretching hard to the left, "
        "warm orange-red on every surface, deep purple shadow in any depression, "
        "sky transitioning from pale blue at zenith to deep amber at the horizon",
        "wind carrying fine dust, a distant hawk, the creak of the dead tree, "
        "and total silence underneath everything — the silence of a landscape indifferent to people"),

    "🍺 Frontier saloon — dusk interior": (
        "interior of a frontier-era saloon, long bar of bare wood on the left, "
        "a mirror behind it age-spotted and dark, bottles in uneven rows, "
        "six or seven tables with mismatched chairs, sawdust on the floor, "
        "a piano in the far corner, a staircase to rooms above, "
        "wanted posters on the wall beside the door, dust motes in the late light",
        "late sun through two windows — long amber shafts cutting through dust, "
        "oil lamp practicals on the bar already lit against the coming dark, "
        "deep shadow in the corners and beneath the staircase",
        "an upright piano in the corner playing a ragtime waltz — slightly out of tune on the high strings, "
        "the pianist visible only as a silhouette — someone drinking alone at the bar, "
        "a chair scraping on floorboards, spurs on the wooden floor as someone stands, "
        "a glass set down hard on the bar top, the staircase creaking under descending weight, "
        "and underneath it all the wind outside finding every gap in the timber walls"),

    # ── DREAMCORE / LIMINAL ──────────────────────────────────────────────
    "🛒 Empty shopping mall — fluorescent liminal": (
        "large shopping mall completely empty of people, long corridors of shuttered storefronts "
        "stretching in both directions, the shutters all down and locked, "
        "a few abandoned planters with dead or fake plants, "
        "a central atrium with a dry fountain, escalators running with no one on them, "
        "the carpet slightly different patterns at each junction suggesting years of piecemeal replacement",
        "overhead fluorescent grid — full brightness, slightly blue-white, no shadows anywhere, "
        "the specific flat even light of a space designed for commerce that no longer happens",
        "the escalators' constant mechanical hum, the HVAC cycling, "
        "a distant jingle from a speaker playing to no one, "
        "footsteps that shouldn't be there echoing from somewhere further in"),

    "🏫 School corridor — after hours": (
        "secondary school corridor at night, lockers running the full length of both walls, "
        "some hanging open, one with a torn photo still attached to the inside door, "
        "classroom doors with small rectangular windows, the rooms dark beyond them, "
        "emergency exit sign at the far end the only non-fluorescent light source, "
        "a forgotten backpack on the floor, a classroom door ajar showing empty desks",
        "overhead fluorescent at half — the end nearest the exit sign off, "
        "creating a gradient from lit to near-dark toward the emergency exit's green cast",
        "the fluorescent buzz, a locker door swinging slightly in a draught, "
        "the distant sound of something institutional — a boiler, a clock — "
        "and the specific silence of a building built for noise now completely empty"),

    "🟨 Backrooms — endless yellow corridors": (
        "an infinite office-like corridor of consistent beige-yellow walls and carpet, "
        "no windows, no doors visible, the corridor turning at irregular intervals, "
        "the same carpet pattern repeating indefinitely, "
        "fluorescent panels in the dropped ceiling, some working some not, "
        "a faint wet-carpet smell implied by the visual texture of the aging floor covering, "
        "the horizon of each corridor always the same distance away regardless of movement",
        "flat fluorescent from the ceiling panels — no shadows, no depth cues, "
        "the light slightly yellow-green from the aging panels, uniformly too bright",
        "a low persistent hum from the lighting and from something deeper in the structure, "
        "no echo — the space absorbs sound — "
        "and the sound of footsteps that are yours and also slightly delayed"),

    # ── ACTION / BLOCKBUSTER ──────���──────────────────────────────────────
    "🏙 Rooftop chase — night city": (
        "rooftop of a city building at night, air conditioning units and water tanks "
        "creating obstacles across the flat roof, gravel underfoot, "
        "the edge with its low parapet visible ahead, the city sprawling below and beyond, "
        "the roof of the next building slightly lower and a gap between them, "
        "wet from recent rain, puddles on the flat membrane roof catching city glow",
        "city ambient glow from every direction as orange fill, "
        "cool blue from the night sky above, practical rooftop lights on the equipment, "
        "the edge of the roof backlit by the city below it",
        "city noise rising from below, wind at height, footsteps on gravel carrying clearly, "
        "a helicopter somewhere — its searchlight sweeping — "
        "and the impact sounds of bodies on metal and concrete"),

    "🏭 Industrial warehouse — emergency lighting": (
        "large industrial warehouse interior, steel-frame structure with a high corrugated ceiling, "
        "abandoned equipment and crated goods on wooden pallets creating a maze of cover, "
        "concrete floor with oil stains and painted navigation lines, "
        "a mezzanine level accessible by metal stairs on the far side, "
        "tall narrow windows at ceiling height letting in fractured moonlight",
        "standard lighting failed — emergency strips only at floor level in red, "
        "moonlight through the ceiling windows in diagonal shafts through dust, "
        "torchlight as a moving motivated source, deep shadow between every structure",
        "the metal structure ticking as it cools, "
        "every footstep echoing in the high ceiling space, "
        "something mechanical still running somewhere — a pump, a conveyor — and then stopping"),

    "🛣 Rain-soaked highway — car chase": (
        "a six-lane highway at night, rain heavy enough to reduce visibility to 50 metres, "
        "headlights of other vehicles forming blurred streaks in the wet, "
        "the road surface a sheet of reflected white and amber, "
        "crash barriers on both sides, an overpass ahead, "
        "the subject vehicle threading between slower traffic at high speed",
        "headlight white from every direction reflected in the wet asphalt, "
        "amber sodium from the highway gantries above, "
        "police or pursuit lighting in blue-red in the rear-view mirror",
        "tyre roar on wet tarmac at speed, rain on the roof and windscreen, "
        "the engine at high revs, the blast of air as a vehicle is overtaken, "
        "a distant siren growing closer"),

    # ── COOKING SHOW ───────────────────────────��─────────────────────────
    "����‍🍳 Professional kitchen — service": (
        "commercial kitchen at full service, stainless steel surfaces everywhere, "
        "six burner ranges with active flames, a pass at the far end where plates are assembled, "
        "the section system visible — hot section, cold section, pastry at the back, "
        "multiple cooks in whites moving with practised urgency, "
        "steam rising from multiple pans, heat visible as shimmer above the ranges, "
        "orders called from the pass, the specific controlled chaos of a kitchen at capacity",
        "overhead fluorescent on stainless — hard, bright, clinical, no shadows — "
        "the flames from the burners providing warm orange counter-light from below, "
        "the pass lit separately in clean white for plating",
        "the roar of extractor fans overhead, burner flames under pans, "
        "the call-and-response of the pass, metal on metal, the hiss of liquid hitting a hot pan"),

    "🍳 Home kitchen — morning light": (
        "domestic kitchen in morning light, an island counter in the centre, "
        "a window above the sink showing a garden or street outside, "
        "used chopping board, a few ingredients out on the counter, "
        "a pan on the hob with a tea towel draped nearby, "
        "the specific lived-in quality of a kitchen used every day",
        "natural morning light through the window — soft, directional, warm white — "
        "the window as the key source, shadows soft to the left of everything, "
        "under-cabinet lighting on if it's still early, adding warm fill to the counter",
        "the hob ticking as it heats, the extractor fan at low, "
        "a radio somewhere in the house, the knife on the board, "
        "water coming to the boil"),

    # ── WES ANDERSON ─────────────────────────────────────────────────────
    "🏨 Grand hotel lobby — Wes Anderson": (
        "a grand hotel lobby of the early-to-mid 20th century, perfectly symmetrical from the camera's position, "
        "a long reception desk centred at the far end, two matching staircases curving up on either side, "
        "a chandelier centred in the ceiling, patterned carpet in a geometric repeat, "
        "a bellboy standing perfectly still at the left, an identical one at the right, "
        "framed portraits evenly spaced on the walls, a revolving door centred in the entrance behind camera",
        "warm amber from the chandelier and wall sconces — even, sourceless-feeling, "
        "the light itself part of the symmetry — no shadow falls asymmetrically",
        "a grandfather clock ticking in precise four-four time, the revolving door cycling at the entrance "
        "with its exact pneumatic sweep and click, a telephone on the front desk ringing twice and stopping, "
        "a bellboy's trolley wheels on marble in perfect straight lines, "
        "someone at the piano in the adjacent salon playing something from 1932 in a major key, "
        "the specific hush of a lobby where every sound is permitted but nothing is loud"),

    "🏘 Pastel townhouse street — afternoon": (
        "a street of terraced townhouses each a different pastel colour — "
        "pale yellow, dusty rose, sage green, powder blue — in a repeating sequence, "
        "perfectly maintained window boxes with matching flowers, "
        "a pavement of identical grey cobbles, "
        "a bicycle of a matching pastel colour leaning against a door on the left, "
        "a letter box, a brass knocker, and a doormat all perfectly centred on each door",
        "flat overcast afternoon — no directional shadow, the pastels fully saturated and even, "
        "the colour of each house reading cleanly against the white of the sky",
        "a bicycle bell ringing once at exactly the right moment, a distant tram on its fixed route, "
        "a window opening on the second floor of the sage-green house — precisely — and closing again, "
        "someone practising scales on a woodwind instrument somewhere behind a wall, "
        "the sound of a letterbox closing, footsteps on cobble in a specific rhythm, "
        "and then complete symmetric silence"),

    # ── K-DRAMA ──────────��────────────────────────────────────────────────
    "🌆 Seoul rooftop — dusk golden hour": (
        "rooftop of a Seoul apartment building at dusk, "
        "laundry lines with clothes barely visible in the fading light, "
        "water tanks and ventilation boxes, a small garden of potted plants in one corner, "
        "the city below spreading to every horizon, apartment towers lit in warm evening windows, "
        "the Han River a faint dark band in the mid-distance, "
        "two folding chairs and a small table — recently used",
        "dusk: the last directional light gone, sky a gradient of deep rose to cool indigo at the zenith, "
        "the city's warm amber rising from below like a second horizon, "
        "a street lamp on the access staircase providing the only warm key light",
        "city hum from below, wind at rooftop height carrying K-indie or lo-fi from an open window several floors down, "
        "a distant siren absorbed into traffic, the creak of a laundry line wire, "
        "the specific rooftop silence that sits just above the city's noise floor — "
        "present enough to feel alone, close enough to feel held"),

    "🌸 Cherry blossom park — midday": (
        "a park with cherry blossom trees in full bloom, "
        "petals continuously falling in the light wind, "
        "a stone path through the trees, wooden benches at intervals, "
        "other people visible in soft focus at the edges — couples, families — "
        "the blossom so dense it forms a soft ceiling overhead, "
        "petals accumulating in drifts against the kerb of the path",
        "filtered overhead light through the blossom canopy — soft pink-white, directionless, "
        "everything in the scene faintly lit from above through the petals, "
        "no hard shadows, skin luminous in the diffused light",
        "wind through the blossom — a collective soft rustle — "
        "petals landing on surfaces with barely any sound, "
        "distant park sounds softened by the canopy, someone laughing"),

    "🛋 Modern Seoul apartment — evening": (
        "interior of a modern Seoul apartment, open-plan living and kitchen area, "
        "floor-to-ceiling glass on one wall showing the Seoul skyline at evening, "
        "minimal furniture — a sofa, a low table, a kitchen island in white and grey — "
        "everything clean and considered, a single personal object on the table "
        "suggesting the room is lived in, a glass of water recently placed",
        "evening: the skyline outside providing ambient warm orange glow through the glass, "
        "interior lighting warm and low — a single floor lamp, no overhead lights, "
        "the glass wall doubling every interior light source in its reflection",
        "the city muffled by the glass — a distant siren, traffic below — "
        "the HVAC at low, the specific silence of a well-insulated modern apartment, "
        "and whatever the scene between the people in it generates"),

    # ── NIGHTLIFE / ADULT VENUES ──��──────────────────────────────────────
    "💃 Strip club — main floor": (
        "strip club interior at full operation, a raised centre stage with a brass pole "
        "catching coloured light, mirrored wall behind the stage doubling everything, "
        "leather booths arranged in a horseshoe around the stage, VIP rope section off to one side, "
        "a long bar with backlit shelves of bottles along the far wall, "
        "scattered tables between stage and bar, each with a small candle flickering in red glass, "
        "smoke machine haze hanging at waist height, a DJ booth tucked in the corner",
        "stage wash cycling slow between magenta, violet, and warm amber — hard spots on the pole, "
        "UV strips along the stage edge making white fabric glow, "
        "deep shadow in the booths beyond the stage light spill, "
        "the mirrored wall creating infinite depth behind the performer",
        "bass-heavy RnB or trap at medium volume, ice in glasses, "
        "low conversation from the booths, heels on the stage surface, "
        "the specific sound of a room designed to keep you looking at the centre"),

    "🔒 Private booth — POV": (
        "POV from a man seated in a strip club private booth, "
        "camera locked at seated eye height looking slightly upward, "
        "black leather seat visible at the lower edge of frame, "
        "a curtain of dark velvet or beaded strands half-drawn behind the performer, "
        "the booth is small — the performer fills the frame at arm's length, "
        "a low table to one side with a drink, the main club visible only as blurred colour and movement "
        "through the curtain gap, a small wall-mounted speaker, dim recessed light overhead",
        "single overhead recessed downlight — warm amber, tight pool, directly above the performance space, "
        "everything outside the light pool near-black, "
        "the performer lit from above with strong shadow below the chin and cheekbones, "
        "occasional colour bleed — magenta, blue — leaking through the curtain from the main floor",
        "bass from the main floor muffled through the curtain, "
        "the booth speaker playing its own quieter track, breathing audible at this proximity, "
        "fabric shifting, the creak of leather seating, ice settling in the glass"),

    # ── BEACHES / OUTDOOR SOCIAL ─────────────────────────────────────────
    "🌴 LA beach — Venice / Santa Monica": (
        "Venice Beach boardwalk spilling onto wide flat sand in late afternoon golden hour, "
        "the Pacific glinting hard silver-gold to the horizon, palm trees in a line along the boardwalk, "
        "skaters and cyclists in soft-focus background on the bike path, "
        "muscle beach gym frames visible further down, graffiti walls and vendor stalls along the walk, "
        "lifeguard tower in classic white and red, crowds scattered across the sand — towels, coolers, "
        "someone playing volleyball, the Santa Monica pier and its ferris wheel visible in the distant haze",
        "golden hour California sun — warm, low, directional from the west over the ocean, "
        "long shadows stretching inland, everything backlit and rim-lit, "
        "skin glowing warm, sunglasses catching flare, the specific amber-pink LA light",
        "waves on the shore in steady rhythm, crowd noise from the boardwalk, "
        "a boombox somewhere playing hip-hop, skate wheels on concrete, "
        "seagulls, distant laughter, the Venice Beach energy that never fully quiets down"),

    "🍹 Ibiza pool party — golden hour": (
        "infinity pool at a cliff-edge villa in Ibiza at golden hour, "
        "the Mediterranean spread below in deep blue, white-washed walls and terracotta tiles, "
        "the pool overflowing its edge into the view, DJ setup under a white canopy, "
        "people in the water and on daybeds around the pool, champagne in ice buckets, "
        "string lights not yet lit waiting for dusk, smoke from a grill drifting across",
        "direct golden hour sun from the west — hard, warm, every water droplet catching it, "
        "skin glistening, pool surface a sheet of shifting gold, "
        "white surfaces bouncing light everywhere as natural fill",
        "deep house from the DJ at medium volume, water splashing, laughter, "
        "glasses clinking, the wind off the Mediterranean, "
        "the specific sound of an afternoon that knows it's about to become a night"),

    "🏄 Bondi Beach — bright midday": (
        "Bondi Beach at midday from the promenade level looking down the crescent of sand, "
        "the ocean a vivid turquoise with white breakers rolling in regular sets, "
        "hundreds of people on the sand, surfers in the water, the iconic red and yellow lifeguard flags, "
        "the sandstone headland at each end of the crescent, Norfolk pines along the promenade, "
        "the Icebergs pool visible cut into the rocks at the south end",
        "harsh Australian midday sun — overhead, no shadow relief, high UV, "
        "bleached sand near-white, ocean almost too bright to look at, "
        "everything saturated and high-contrast, sunscreen-sheen on skin",
        "surf crash in steady sets, crowd buzz, lifeguard whistle, "
        "someone's portable speaker, seagulls fighting over chips, "
        "the specific roar of a packed beach at the height of summer"),

    # ── MOODY / CINEMATIC INTERIORS ──────────────────────────────────────
    "🕯 Candlelit loft — exposed brick": (
        "open loft apartment with exposed brick walls and timber ceiling beams, "
        "the only light from clusters of pillar candles — on the floor, on shelves, on a low table, "
        "thirty or forty flames creating overlapping pools of warm amber, "
        "a large bed with dark linen visible in the back half of the space, "
        "a freestanding cast-iron bathtub near the windows, "
        "tall industrial windows showing the city at night but curtained with sheer fabric",
        "candlelight only — warm amber from multiple low sources, "
        "flames creating soft moving shadows on the brick, "
        "the candles reflected in the dark window glass, deep shadow above the beam line",
        "candle flames guttering in a draught, distant city through the glass, "
        "the creak of old timber, fabric shifting, "
        "the specific intimate quiet of a room lit only by fire"),

    "🚿 Rain shower — glass-walled bathroom": (
        "large walk-in rain shower with floor-to-ceiling glass walls on two sides, "
        "a single oversized showerhead directly overhead raining straight down, "
        "steam filling the upper half of the glass enclosure, "
        "water streaming in sheets down the glass, "
        "dark slate tile floor and walls, recessed warm LED strip at floor level, "
        "a bench built into the back wall, the bathroom beyond the glass visible but soft through steam",
        "recessed warm LED strip at floor level casting upward through the steam and water, "
        "overhead downlight diffused through the rain and mist, "
        "everything soft-edged and glowing, skin wet and catching every light source",
        "rain shower hiss from directly overhead — enveloping, constant, "
        "water hitting slate, steam, breathing amplified by the glass enclosure, "
        "the specific acoustics of a tiled glass box"),

    "🪩 Hotel rooftop bar — city night": (
        "rooftop bar on a high-end hotel, the city skyline as the backdrop on three sides, "
        "the bar itself a long backlit slab of marble or onyx, cocktails in progress, "
        "low seating clusters — velvet and brass — arranged around fire pit tables, "
        "a small pool or water feature reflecting the city lights, "
        "well-dressed people at the edges, a DJ playing from a minimal booth, "
        "string lights and pendant fixtures overhead creating warm islands of light",
        "warm practical lighting from the bar, fire pits, and string lights, "
        "city skyline ambient glow as backdrop, "
        "the sky a deep dark blue with the city preventing true black",
        "cocktail bar sounds — shaker, ice, glass on marble, low conversation, "
        "deep house at low volume from the DJ, wind at height, "
        "the city far below as a continuous ambient hum"),

    # ── TRANSPORT / MOTION ────────��──────────────────────────────────────
    "🛥 Yacht deck — open ocean sunset": (
        "aft deck of a motor yacht at sunset, teak deck underfoot, "
        "the wake stretching back white and straight to the horizon, "
        "open ocean in every direction — deep blue turning to copper near the sun, "
        "the stern rail and a pair of chaise lounges, champagne in a bucket lashed to the rail, "
        "the upper flybridge visible above casting a shadow across the back half of the deck, "
        "sea spray occasionally reaching the lower deck",
        "direct sunset from the stern — warm copper-gold, hard rim light on everything facing aft, "
        "deep blue shadow on the forward side, the wake itself catching the light, "
        "skin lit warm from behind, face in soft reflected ocean fill",
        "engine vibration through the deck, wind, the hull cutting water, "
        "wake turbulence behind, a halyard clinking somewhere, "
        "the deep isolation of being the only thing on the ocean"),

    "🏎 Supercar interior — night drive": (
        "interior of a low-slung supercar at night — Lamborghini, McLaren, or similar — "
        "the cockpit tight and low, carbon fibre dash and centre console, "
        "the instrument cluster glowing warm amber behind the flat-bottom steering wheel, "
        "city lights streaking past through the low windshield, "
        "LED ambient strips along the door sills in cool blue, the seats deep bucket-shaped, "
        "the road surface visible through the windshield blurred with speed",
        "instrument cluster glow from below — warm amber, "
        "LED ambient strips in cool blue along the sills, "
        "city light streaking through the glass in rhythmic pulses, "
        "the driver's face lit from below and from the passing city",
        "engine note — a specific high-RPM mechanical scream behind and below the seats, "
        "tyres on asphalt, wind noise at speed, "
        "the turbo spool between shifts, city sound entering and leaving in doppler pulses"),

    # ── RAW / GRITTY ──────────────────────────���──────────────────────────
    "🏨 Cheap motel room — neon through blinds": (
        "single-room motel interior at night, a queen bed with a thin patterned bedspread, "
        "wood-veneer furniture, a CRT TV on the dresser, venetian blinds at the window "
        "casting horizontal neon stripes — red and blue — across the bed and opposite wall, "
        "the bathroom door ajar showing harsh fluorescent inside, "
        "a bag on the floor, car headlights occasionally sweeping across the ceiling",
        "neon from outside through the blinds — alternating red and blue in horizontal bands, "
        "harsh bathroom fluorescent spilling through the cracked door as a single cold stripe, "
        "headlight sweeps across the ceiling at irregular intervals, "
        "the room itself has no light on — everything lit from outside or the bathroom",
        "the neon sign buzzing outside the window, ice machine humming through the wall, "
        "distant traffic on the highway, a door slamming somewhere in the building, "
        "the specific acoustic of thin walls and a parking lot outside"),

    "🏗 Industrial warehouse — night": (
        "cavernous warehouse interior at night, concrete floor cracked and oil-stained, "
        "steel columns running in a grid to the far wall, high corrugated roof lost in shadow, "
        "a few industrial pendant lights still working casting hard pools on the floor, "
        "loading dock doors along one wall — one rolled halfway up showing the dark yard outside, "
        "a car parked inside with its headlights on cutting two beams through the dust",
        "hard pools of light from the industrial pendants — warm sodium orange, "
        "car headlights cutting white beams through floating dust, "
        "deep black shadow between the light pools, the roof invisible",
        "echo — everything echoes in here, footsteps, voices, the drip from a pipe, "
        "a distant generator running, wind through the half-open loading dock, "
        "the specific reverb of a concrete box fifty metres long"),

    # ── RURAL / EQUESTRIAN ────���──────────────────────────────────────────
    "🐴 Horse stable — warm afternoon": (
        "centre aisle of a large horse stable, stalls lining both sides with wooden half-doors, "
        "horses visible in several stalls — heads over the doors, ears forward, watching, "
        "the aisle floor compacted earth and straw, hay bales stacked against the far wall, "
        "tack and bridles hanging from iron hooks between stalls, "
        "afternoon light streaming through the open barn doors at the far end "
        "in long golden shafts full of floating dust and hay particles, "
        "the timber roof beams high overhead with swallows nesting in the crossbeams",
        "warm directional afternoon sun from the open barn doors — long golden shafts cutting the aisle, "
        "the stalls in warm shadow, straw on the floor catching the light, "
        "dust motes and hay particles suspended in every beam of light, "
        "deep amber warmth throughout, cool shadow in the stalls themselves",
        "horses shifting weight in their stalls — hooves on straw, a snort, "
        "a tail swishing against wood, the creak of a stall door, "
        "swallows above, distant meadow sounds from outside the barn, "
        "the deep quiet underneath everything that says countryside"),

    "🐴 Horse stable — night lantern": (
        "horse stable at night, the aisle lit by a single hanging lantern "
        "swaying gently from a roof beam, casting moving amber light and shadow, "
        "stalls on both sides — horses dozing, one head visible over a door, "
        "straw deep on the aisle floor, a saddle resting on a stand by the far wall, "
        "the barn doors closed against the dark, a gap at the top showing stars, "
        "a wool blanket folded on a hay bale, the smell of horse and leather implied by every surface",
        "single hanging lantern — warm amber, swaying, casting moving shadows "
        "that shift across the stall doors and the roof beams, "
        "everything beyond the lantern's reach in deep warm darkness, "
        "the horses' eyes catching the light from inside their stalls",
        "a horse breathing slow and heavy in the nearest stall, straw rustling, "
        "the lantern chain creaking with its sway, a horse stamping once, "
        "wind outside the closed doors, an owl somewhere beyond the barn, "
        "the complete rural silence that makes every small sound distinct"),

    "🌾 Barn interior — hay loft": (
        "upper hay loft of a large timber barn, the floor thick with loose hay and straw, "
        "a loft door open to the countryside showing fields stretching to the horizon, "
        "the roof beams close overhead — rough-hewn timber, iron bolts, cobwebs, "
        "bales stacked against the back wall, a pitchfork leaning in the corner, "
        "the loft edge with a wooden rail looking down to the barn floor below, "
        "golden late-afternoon light flooding through the open loft door",
        "golden hour sun pouring through the open loft door — directional, warm, "
        "every piece of hay in the air backlit and glowing, "
        "the light hitting the loose straw on the floor and turning it to gold, "
        "deep shadow against the back wall behind the bales",
        "wind through the open loft door, hay shifting, "
        "birds in the rafters, distant farm sounds — a tractor, a dog, "
        "the creak of the old timber structure, the countryside beyond the door"),

    "🏡 Farmhouse kitchen — early morning": (
        "large farmhouse kitchen at dawn, an Aga or wood-burning range against one wall "
        "radiating warmth, a scrubbed pine table in the centre with mismatched chairs, "
        "a window over the sink showing fields in early mist, "
        "copper pans hanging from a ceiling rack, a stone floor with a woven rug, "
        "a collie asleep in a basket by the range, a mug of tea steaming on the table",
        "cold blue dawn light through the window mixing with warm orange from the range, "
        "the two colour temperatures meeting in the middle of the kitchen, "
        "her face lit warm from one side and cool from the other",
        "the range ticking as it heats, a clock on the wall, "
        "birdsong building outside, the dog breathing in its basket, "
        "a kettle not yet boiling, the specific deep quiet of a farmhouse before the day starts"),

    # ── EXPERIMENTAL — ULTRA DETAIL ─────��────────────────────────────────
    "🚀 [EXPERIMENTAL] Rocket launch pad — close range countdown": (
        "launch pad complex at T-minus 5 seconds, the rocket a 70-metre column of white-painted steel "
        "and composite panels rising directly in front of the camera at a distance of 300 metres, "
        "close enough that the full rocket does not fit in frame — "
        "the camera is angled upward capturing the lower third of the vehicle: "
        "the engine cluster, the launch mount arms still clamped at the base, "
        "the flame trench below filled with the water suppression system in full flow — "
        "a white curtain of steam already billowing upward and outward from the base, "
        "the rocket body showing condensation streaks from the cryogenic propellant "
        "running down the pale exterior in irregular rivulets, "
        "launch mount service arms still attached at multiple levels — "
        "each arm a steel structure 3 metres wide with utility connections and umbilical feeds, "
        "the hold-down bolts visible at the base still engaged, "
        "at T-3 the engine ignition sequence begins — "
        "a pale blue-white flame appears at the base of the engine cluster, "
        "not yet at full thrust, building in a rapid sequence visible as a brightening bloom "
        "that lights the steam cloud from within, "
        "at T-0 the hold-down bolts release and the full engine thrust registers — "
        "the steam cloud erupts outward in every direction from the flame trench, "
        "the rocket lifts — slowly at first, the enormous mass requiring two full seconds "
        "to clear the launch tower, the service arms swinging away, "
        "the engine exhaust plume expanding below as the vehicle accelerates, "
        "the ground shaking visible as camera vibration, "
        "debris — small gravel, dust, paper — lifting off the ground around the camera position, "
        "the sky above the rocket clearing to a deep blue as the vehicle climbs",
        "pre-launch: harsh white xenon floodlights from the launch tower illuminating the rocket "
        "in cold clinical light against a pre-dawn dark blue sky, "
        "T-0: the engine ignition creating its own light source — "
        "a blue-white core at 3500 degrees transitioning to orange at the plume edges, "
        "the entire scene converting from flood-lit industrial to fire-lit in under one second, "
        "the steam cloud lit orange from within as the plume expands through it, "
        "everything above the plume still in pre-dawn blue while everything below is orange-white fire",
        "the countdown from an unseen speaker — each number distinct and flat, "
        "the water suppression system as continuous white noise building in volume, "
        "at ignition a sound that arrives as physical pressure before it arrives as audio — "
        "a crackling roar that builds from a distant rumble to a full chest-compression event "
        "in under three seconds, the ground vibration arriving through whatever surface the camera contacts, "
        "the steam cloud hissing, the hold-down release as a mechanical crack lost in the engine noise, "
        "and after the vehicle clears the tower the sound continuing to build "
        "as the full plume establishes and the rocket accelerates away"),

    "🚕 [EXPERIMENTAL] Fake taxi — parked, discrete location": (
        "interior of a taxi cab parked in a quiet layby or side street, engine off, "
        "the vehicle a standard four-door sedan with a taxi livery — "
        "yellow or black depending on city, a roof sign unlit since the meter is off, "
        "back seat wide enough for two with worn dark fabric upholstery, "
        "a cigarette burn on the left armrest, a pine air freshener hanging from the rear-view mirror, "
        "a partition of scratched perspex between front and back seat "
        "with a small sliding cash window currently open, "
        "the driver turned around in the front seat with one arm over the headrest, "
        "facing the back, "
        "the vehicle is stationary — no road movement, no engine vibration, "
        "parked somewhere deliberately quiet: a darkened layby off a main road, "
        "a side street behind commercial buildings, a car park with one working light at the far end, "
        "the windows fogging slightly from body heat inside the sealed car, "
        "occasional headlights from passing traffic sweeping through the rear window "
        "and crossing the interior before disappearing, "
        "the city or countryside outside muffled and indifferent through the closed doors, "
        "the back seat functionally private — no pedestrians, no other vehicles stopped nearby, "
        "the taxi meter display dark on the dashboard, "
        "a dashcam mounted on the windscreen, its small red recording light visible in the mirror",
        "ambient light from outside the parked vehicle — "
        "a distant streetlamp providing a low amber fill through the rear and side windows, "
        "headlights from passing traffic creating sweeping white flares at irregular intervals "
        "that fully illuminate the interior for half a second before returning to dim amber, "
        "the dashcam LED a small constant red point in the upper frame, "
        "the partition perspex catching light and creating a faint reflection of the back seat "
        "visible to the driver — and to the camera",
        "near silence of a parked vehicle in a quiet location — "
        "the engine cooling with irregular metallic ticks, "
        "distant road traffic as a low continuous presence that rises when a vehicle passes close "
        "and fades back to nothing, "
        "the slight creak of the suspension under shifting weight, "
        "fabric moving against leather and fabric, "
        "the pine freshener swinging against the mirror on any movement, "
        "and the specific sealed acoustic of a car interior "
        "where every sound is close and contained"),

    "🚁 [EXPERIMENTAL] Flying car interior — neon megalopolis night": (
        "interior of a luxury flying car cockpit suspended 800 metres above a sprawling megalopolis at 2am, "
        "the canopy glass a seamless wraparound bubble giving unobstructed 270-degree views of the city below, "
        "every direction filled with other flying vehicles at different altitudes — delivery drones in tight formation lanes, "
        "heavy freight barges with blinking amber warning lights drifting slowly through the mid-tier, "
        "sleek personal vehicles weaving the upper express corridors in streaks of white and red light, "
        "the city surface 800 metres below is a continuous carpet of neon — magenta, cyan, gold, white — "
        "interrupted by the dark canyons between tower blocks that plunge into unlit depths, "
        "holographic advertising pillars rising from rooftops project rotating brand logos into the low cloud layer, "
        "rain is hitting the canopy glass constantly, each droplet refracting the city below into smeared colour streaks "
        "that run sideways as the vehicle banks, the interior of the cockpit is tight and deliberately minimal — "
        "a curved dashboard of brushed obsidian inlaid with haptic control surfaces glowing in soft amber and blue, "
        "the pilot seat in worn dark leather with silver stitching, a cracked personal screen mounted centre showing "
        "navigation overlays and atmospheric warning data in thin white lines, "
        "the floor is a single pane of clear glass revealing the city below through the undercarriage, "
        "turbulence causes the vehicle to shudder at irregular intervals, "
        "the city towers on either side are close enough to read the wear on their cladding — "
        "oxidised copper panels, exposed concrete poured in the previous century, "
        "retrofitted thermal insulation in grey foam blocks strapped with galvanised bands, "
        "window units of a thousand apartments stacked in irregular grids, some lit warm, most dark, "
        "a maintenance worker in a harness working on an external unit three floors down on the nearest tower, "
        "visible for two seconds before the vehicle passes",
        "city ambient glow from below as the dominant light source — a shifting mix of magenta, cyan, and sodium amber "
        "washing upward through the canopy glass and painting everything inside in moving colour, "
        "the dashboard instruments providing a secondary warm amber fill from below, "
        "no overhead light — the cockpit interior is lit entirely by the city and the controls, "
        "rain on the canopy refracting every light source into moving prismatic smears across the pilot's face and hands, "
        "when the vehicle banks the lighting shifts completely as different colour zones of the city pass beneath",
        "the constant high-frequency white noise of the city at this altitude — not traffic but the aggregate of "
        "ten million sound sources filtered through 800 metres of air into a single undifferentiated pressure, "
        "the vehicle's own turbines as a low directional vibration felt more in the seat than heard, "
        "rain hammering the canopy glass in irregular bursts as wind speed changes, "
        "proximity alert tones from the navigation system as vehicles pass within 50 metres, "
        "the creak of the cockpit frame flexing in turbulence, "
        "and through the communication channel a distant air traffic controller voice reading coordinates "
        "in a flat monotone that cuts out mid-sentence"),

    "🌆 [EXPERIMENTAL] Neon megalopolis street — midnight rain": (
        "ground level on the main commercial boulevard of a future megalopolis at midnight, "
        "the street is 40 metres wide and lined on both sides by towers that rise out of frame overhead, "
        "their faces covered floor-to-ceiling in LED advertising panels that cycle through product imagery "
        "in saturated colour — a perfume ad in 30-metre-tall slow motion, a food delivery brand in "
        "rapid-cut animation, a political message in white text on red cycling every four seconds, "
        "holographic projections extend from building facades into the street itself — "
        "a 15-metre translucent woman walks alongside foot traffic without interacting, "
        "a brand logo rotates slowly at intersection height, casting coloured light on wet pavement below, "
        "the pavement is packed — bodies moving in every direction at different speeds, "
        "delivery workers on electric cargo bikes threading through gaps, "
        "street vendors with lit carts selling food and counterfeit hardware from fixed positions, "
        "security drones at 5-metre altitude patrolling slow circuits above the crowd, "
        "a busker 20 metres ahead performing with a live instrument amplified through a cracked speaker stack, "
        "the street surface is wet from rain that stopped 20 minutes ago — "
        "every neon reflection doubled in the standing water on the pavement, "
        "gutters running with grey water carrying food packaging and disposable packaging east toward the drain grid, "
        "steam venting from three separate grate locations in irregular pulses, "
        "the smell of cooking meat, hot circuit boards, and ozone from the drone systems "
        "implied by the visual density of the scene, "
        "overhead the transit rail runs on a concrete viaduct 12 metres up, "
        "a train passing every 90 seconds and throwing sparks from the contact rail that drift down "
        "through the advertising light and land on the crowd as brief orange points",
        "total colour saturation from every direction simultaneously — "
        "no single dominant source, light arriving from left right above and reflected from below, "
        "the palette cycling constantly as the advertising panels change — "
        "one moment the street is washed magenta, four seconds later white, then cyan, then deep red, "
        "the holographic projections casting translucent coloured fill that passes through solid objects, "
        "wet pavement doubling every source in rippling reflection, "
        "the underside of the transit viaduct a deep shadow that swallows everything above head height "
        "until the next train passes and throws sparks",
        "the city as pure undifferentiated sound pressure — traffic, crowd, music, advertising audio "
        "from multiple competing speakers on different cycles, drone motor harmonics, "
        "the transit rail above — a rising electric whine building to thunder then gone, "
        "sparks falling silent on wet pavement, the vendor nearest camera calling out in two languages, "
        "and under everything the low 60hz hum of the power infrastructure feeding the advertising grid"),

    "🛸 [EXPERIMENTAL] Zero-gravity space station — interior hub": (
        "interior of a large rotating space station hub module in low Earth orbit, "
        "the module is cylindrical, 30 metres in diameter and 60 metres long, "
        "the curvature of the floor visible — the far end of the module curving upward and overhead "
        "so that standing at one end you can see people working on what appears to be the ceiling "
        "but is simply the far section of the curved floor, "
        "the station is old — panels on every surface show decades of use, "
        "thermal blanket insulation patched with silver tape at the seams, "
        "cable bundles running exposed along the walls secured with plastic clips every metre, "
        "handhold rails bolted at 1.5-metre intervals across every surface including the ceiling, "
        "equipment racks bolted floor to ceiling holding grey equipment boxes with status LEDs, "
        "three large circular viewport windows at mid-module showing the curvature of Earth below — "
        "the Indian Ocean in deep blue with a cyclone system visible in the southern hemisphere, "
        "the terminator line visible at the right edge of the viewport where day becomes night, "
        "floating objects throughout the space — a stylus rotating slowly in the middle distance, "
        "a coffee pouch spinning end over end near the ceiling, "
        "a clipboard with attached pen drifting past a work station, "
        "two crew members in grey flight suits working at stations on what is locally their floor "
        "but appears from camera to be the curved side wall, "
        "every loose object secured with velcro or tether clips, "
        "the scale of everything slightly wrong — storage hatches positioned for zero-g reach "
        "rather than standing-human ergonomics, the lighting strips positioned for 360-degree coverage "
        "because there is no single down, condensation visible on the viewport glass inner surface "
        "collecting in small spheres that drift off the glass when disturbed",
        "fluorescent strip lighting running the full length of the module in four parallel lines "
        "positioned at 90-degree intervals around the circumference — even, clinical, "
        "casting no shadows because light arrives from every direction simultaneously, "
        "the Earth through the viewports providing a shifting blue ambient that changes "
        "as the station rotates — one full rotation every 90 minutes cycling from sunlit blue "
        "to orbital night black and back, "
        "equipment indicator LEDs providing small points of green amber and red throughout the space",
        "the specific sound of a pressurised environment — the constant cycling of the air system "
        "as a low directional rush that changes character depending on which vent is nearest, "
        "the structure ticking and groaning as it passes from sunlit to shadow in the thermal cycle, "
        "equipment cooling fans at slightly different frequencies creating a slow beat pattern, "
        "the velcro sound of someone repositioning a tether, "
        "and underneath everything the profound quiet of a sealed environment "
        "with 400 kilometres of vacuum on the other side of 12mm of aluminium"),

    "🌊 [EXPERIMENTAL] Monsoon flood market — Southeast Asia night": (
        "a traditional covered market in a Southeast Asian city at the peak of monsoon season, "
        "the market is a permanent structure — a steel roof on painted concrete pillars spanning "
        "an area the size of a city block, beneath it 200 vendor stalls in irregular rows "
        "selling produce, cooked food, clothing, electronics, and hardware, "
        "the floor is currently underwater — 30 centimetres of brown flood water covering the entire market floor, "
        "the water moving in a slow current from the north end toward the drainage channels at the south, "
        "carrying with it floating packaging, a flattened cardboard box, leaves, and an empty plastic bottle "
        "slowly rotating as it drifts, "
        "vendors have responded to the flooding by elevating their displays — "
        "produce stacked on the highest shelf of their trolleys, "
        "electronics wrapped in plastic bags and raised on wooden crates, "
        "clothing hung from the roof structure above the flood line, "
        "customers and vendors moving through the flood water on foot — "
        "some in rubber sandals, some barefoot, some having removed shoes and tied them to their bags, "
        "the water disturbed into spreading circles and V-shaped wakes by every footstep, "
        "a food vendor at a propane-powered wok is still cooking — "
        "the wok stand raised on two concrete blocks above the water line, "
        "the flame burning blue-orange underneath, steam and smoke rising into the roof space, "
        "the smell of frying garlic and chilli implied by the visual of the smoke direction and density, "
        "rain audible on the steel roof as continuous white noise that changes pitch with wind gusts, "
        "the roof has three leaks — water falling in heavy columns at intervals between the stalls, "
        "the largest leak has had a plastic bucket placed under it that is already overflowing, "
        "a cat is sitting on top of the highest shelf of a dry goods stall, watching the water",
        "fluorescent tube lighting hanging from the roof structure on cables — "
        "some working, some flickering, two dark, "
        "the working tubes reflecting as white bars in the flood water below, "
        "the wok fire providing a moving warm orange source that casts the nearest vendor in flickering fill, "
        "the rain on the roof diffusing sound into a grey-white ambient that the fluorescent light cuts through "
        "in clinical tubes, outside the market visible as total darkness and rain",
        "the steel roof under monsoon rain — a physical presence of sound, "
        "not background but foreground, varying from steady drum to percussive hammering as wind drives harder rain, "
        "flood water being disturbed by footsteps in irregular splashes and waves, "
        "the propane wok hissing and spitting, the vendor calling prices over the rain, "
        "a generator somewhere under the market running the lights in a low mechanical pulse, "
        "and the three roof leaks each hitting their collection points in different rhythms — "
        "bucket, concrete, open water — three distinct pitches of the same water"),

    "🌋 [EXPERIMENTAL] Active volcano observatory — eruption event": (
        "a volcanic observatory research station built on the stable flank of an active stratovolcano, "
        "the station a collection of reinforced concrete and steel structures bolted to basalt bedrock "
        "at 2,400 metres altitude, the main observation deck a steel-grate platform with a welded railing "
        "extending from the primary building over a 200-metre drop to the lava field below, "
        "the volcano is in active eruption — the summit crater 800 metres above the station "
        "is continuously ejecting material: lava fountains visible as orange-red columns against the night sky, "
        "pyroclastic ejecta — rocks ranging from fist-sized to car-sized — "
        "rising in slow arcs and falling in the illuminated zone around the crater, "
        "the lava field below the station is active — new lava moving in a slow viscous river "
        "across older cooled black basalt, the active flow glowing orange at its leading edge "
        "and fading to dark red further back where cooling has begun, "
        "the air above the lava field is visibly distorted by heat shimmer, "
        "sulfur dioxide gas visible as a yellowish haze in the middle distance, "
        "ash fall is continuous — fine grey-black particles accumulating on every horizontal surface "
        "at 2-3mm per hour, the observation deck railing has a visible ash line on its upper edge, "
        "the wind direction is shifting — ash coming directly toward the camera in one gust "
        "then cutting off as the wind rotates, "
        "the station building behind the deck has blast-proof shutters on all windows, "
        "most closed, one partially open showing a lit interior with monitoring equipment screens, "
        "a seismic drum recorder visible through the gap, its needle moving in continuous tight oscillation, "
        "on the observation deck itself: a researcher in a hard hat, respirator, and heat-resistant suit "
        "is operating a thermal imaging camera on a tripod, "
        "securing the tripod against wind gusts with both hands between measurements, "
        "the basalt rock surface of the deck is warm underfoot — "
        "residual heat from the lava field conducting upward through the mountain",
        "the volcano as the dominant light source — "
        "the crater illumination casting an orange-red wash that varies in intensity "
        "with each new fountain pulse, light arriving from above and to the left, "
        "hard shadows shifting as the eruption intensifies and fades in irregular cycles, "
        "the lava field below providing a secondary orange fill that rises from beneath "
        "and lights the underside of ash clouds drifting across the mid-level, "
        "the overall palette deep black and ash-grey cut through with orange-red from every volcanic source, "
        "lightning visible in the eruption column above the crater — volcanic lightning, "
        "a brief white-blue flash that illuminates the full ash cloud for a fraction of a second",
        "the eruption as physical sound — not a single event but a continuous layered phenomenon: "
        "a deep sub-bass rumble felt in the chest and conducted through the station floor as vibration, "
        "above that the intermittent artillery crack of larger ejecta leaving the crater, "
        "the hiss and roar of gas venting through the crater rim in sustained jets, "
        "closer: the specific sound of lava moving — a slow viscous tearing as the flow advances "
        "over older rock, occasional sharp cracks as cooled crust breaks under the advancing front, "
        "the ash fall on the deck as a near-silent continuous hiss, "
        "wind gusting through the station structure and the railing producing a changing pitch, "
        "and the researcher's respirator — the mechanical rhythm of filtered breath "
        "audible in the brief pauses between eruption pulses"),

    # ── HISTORICAL ──────────────────────────────────────────────────────
    "🕯 Victorian parlor — gaslight afternoon": (
        "high-ceilinged parlor with dark walnut wainscoting rising to shoulder height, "
        "heavy velvet drapes in deep burgundy pulled half-open at tall sash windows, "
        "a Persian rug in faded crimson and navy covering most of the hardwood floor, "
        "oil paintings in ornate gilded frames lining the upper walls, "
        "a marble fireplace with a brass fender, antimacassars on every chair, "
        "lace curtains diffusing the afternoon light into soft white panels",
        "warm afternoon daylight entering through lace curtains — soft, diffused, slightly golden, "
        "gaslamp sconces on the walls providing a secondary amber fill from both sides, "
        "deep shadows pooling behind the heavy furniture and in the folds of the drapes, "
        "the overall palette warm mahogany and cream with gold accents from the gas flames",
        "the faint hiss of gas mantles on the wall sconces, a carriage clock ticking on the mantelpiece, "
        "the muffled clatter of a horse-drawn cab passing on cobblestones outside, "
        "fabric rustling as someone shifts in a wingback chair, "
        "the distant clang of a servant's bell from deeper in the house"),

    "⚔ Medieval castle — torchlit stone": (
        "soaring great hall of a medieval castle, rough-hewn limestone walls rising three storeys "
        "to massive dark oak beams spanning the vaulted ceiling, "
        "a long wooden trestle table running the centre of the hall, scarred with use, "
        "iron torch sconces bolted into the stone at irregular intervals, "
        "a cold flagstone floor with rushes scattered underfoot, "
        "tapestries hanging on the far wall depicting hunting scenes, faded and smoke-darkened",
        "torchlight from iron sconces — warm orange with a strong flicker, "
        "each torch casting its own moving shadow that shifts with every draught, "
        "deep black shadow between the pools of light, the ceiling lost in darkness above the beam line, "
        "torch smoke curling upward in visible ribbons caught by the updraught",
        "the crack and spit of burning pitch in the torches, wind finding gaps in the shuttered windows, "
        "a deep ambient echo off the stone walls turning every sound into a trailing reverb, "
        "distant metallic clang of a gate or chain somewhere in the keep, "
        "the rush of air through the chimney flue at the far end of the hall"),

    "🍷 1920s speakeasy — jazz age": (
        "underground speakeasy hidden behind a false wall, low ceiling with exposed pipes, "
        "dim red velvet booths lining the walls, a polished mahogany bar with a brass foot rail, "
        "shelves of amber-brown bottles backlit behind the barkeep, "
        "cigarette smoke layering the air in visible blue-grey strata, "
        "Edison bulbs hanging from cloth-wrapped cords casting small warm pools of light, "
        "a small raised stage in the corner with a drum kit and upright bass",
        "warm amber Edison bulbs as primary light, each one a visible hot filament, "
        "deep red tones from the velvet upholstery reflecting into the surrounding shadow, "
        "smoke catching every light source and turning beams visible, "
        "the overall palette dark — mahogany, burgundy, brass, and cigarette haze",
        "a jazz trio playing in the corner — upright bass, muted trumpet, brushes on a snare, "
        "low conversation and occasional laughter, ice clinking in heavy tumblers, "
        "the sharp pop of a cocktail shaker, a match striking, "
        "and beneath it all the muffled thrum of the city above filtering through the ceiling"),

    "🏛 Ancient Rome forum — marble columns": (
        "the Roman Forum at midday, massive fluted marble columns rising in colonnades, "
        "white travertine stonework blazing in the direct sun, "
        "sharp black shadows slicing across the paved ground in geometric patterns cast by the columns, "
        "distant temples visible through the colonnade with their triangular pediments, "
        "a broad open plaza with stone steps descending to a lower level, "
        "inscriptions carved into every surface, pigeons on the upper entablatures",
        "direct overhead noon sun — nearly shadowless on horizontal surfaces, "
        "but the columns and porticoes creating hard-edged vertical shadows in parallel stripes, "
        "the white marble acting as a reflector bouncing fill light into every recess, "
        "the palette bleached white and pale gold with deep blue sky above",
        "the acoustic space of open stone — every sound reflecting sharply off marble, "
        "distant crowd murmur echoing across the plaza, footsteps on stone, "
        "pigeons taking flight in a sudden clatter of wings, "
        "wind channelling between the colonnades in a low sustained note"),

    # ── SCI-FI ──────────────────────────────────────────────────────────
    "🛸 Cyberpunk street market — neon rain": (
        "dense street market at night crammed into a narrow alley between tower blocks, "
        "holographic signage in Mandarin, Korean, and English flickering above vendor stalls, "
        "corroded metal awnings extending over tables of electronics and street food, "
        "wet pavement reflecting RGB neon in smeared colour streaks — magenta, cyan, amber, "
        "cables and conduits strung overhead in tangled bundles between buildings, "
        "steam rising from grills mixing with light rain in the neon glow",
        "neon as the dominant light source — no natural light, no sky visible, "
        "holographic projections adding shifting coloured fill that changes with each sign cycle, "
        "wet surfaces doubling every light source in distorted reflections, "
        "deep shadow in every gap between stalls, the overall palette saturated and high-contrast",
        "rain on metal awnings in a continuous irregular drumming, "
        "vendors calling out in overlapping languages, sizzling oil from food stalls, "
        "electronic music bleeding from a doorway, the hum of transformers and neon ballasts, "
        "a police drone passing overhead with a low mechanical whine"),

    "🔬 Sci-fi research lab — clean room": (
        "sterile research lab with smooth white walls and seamless floor, "
        "transparent barrier walls dividing the space into containment zones, "
        "holographic displays floating above minimalist workstations showing data streams and molecular models, "
        "robotic arms suspended from ceiling tracks in standby position, "
        "a specimen chamber at the centre of the room glowing faintly from internal illumination, "
        "every surface immaculate, no dust, no fingerprints, no imperfection",
        "soft diffused blue-white LED panels embedded in the ceiling providing even shadowless illumination, "
        "the holographic displays adding subtle cyan and green accents to nearby surfaces, "
        "the specimen chamber casting a warm amber glow outward in a contained radius, "
        "the overall palette clinical white with cool blue undertones",
        "near-silence — the soft hum of air filtration cycling continuously, "
        "a faint high-pitched tone from the holographic projectors, "
        "the occasional click and whir of a robotic arm repositioning, "
        "the sealed environment creating a pressure in the ears — the sound of absolute sterility"),

    "🌌 Mars research base — red dust": (
        "interior of a pressurised dome on Mars, curved transparent ceiling panels revealing "
        "a rust-red landscape of low ridges and dust under a butterscotch sky, "
        "monitoring equipment banked along the perimeter walls with status lights blinking, "
        "a central work area with geological samples in sealed containers, "
        "dust coating the lower edges of every window despite air filtration, "
        "a pressure suit hanging on a rack near the airlock door, helmet visor reflecting the room",
        "filtered Martian daylight entering through the dome — a diffused amber-pink, "
        "supplemented by cool white interior LEDs on adjustable arms at each workstation, "
        "the red dust on the windows tinting all natural light with a warm ochre cast, "
        "status LEDs adding small points of green, amber, and red across the equipment banks",
        "the constant background hum of life-support systems — air recyclers, pressure regulators, "
        "a rhythmic clicking from a radiation monitor near the airlock, "
        "wind-blown dust ticking against the dome panels in irregular bursts, "
        "the muffled howl of a Martian wind audible as a low-frequency vibration through the structure"),

    # ── SPORTS ──────────────────────────────────────────────────────────
    "🥊 Boxing ring — under spotlights": (
        "professional boxing ring seen from just outside the ropes at canvas level, "
        "white canvas marked with scuffs, resin dust, and faint stains from previous bouts, "
        "four corner posts with turnbuckle pads in red and blue, "
        "three ropes on each side catching the overhead light in parallel lines, "
        "the crowd beyond the ring visible only as a dark mass of silhouettes and occasional camera flashes, "
        "a ring-side bell and water bucket visible in the nearest corner",
        "hard overhead spotlights from a lighting rig directly above the ring, "
        "creating intense top-down illumination with sharp shadows beneath the ropes and corner posts, "
        "the canvas blazing white under the lights while everything beyond the ring falls to near-black, "
        "the geometric pattern of rope shadows crossing the canvas in parallel stripes",
        "crowd noise — a constant roar rising and falling with the action, "
        "the sharp slap of leather on skin, heavy breathing and the shuffle of feet on canvas, "
        "a cornerman shouting instructions between the ropes, "
        "the ring-side bell cutting through everything with a single clean metallic tone"),

    "🏀 Basketball court — arena floodlights": (
        "basketball court from floor level looking across the polished hardwood, "
        "clear painted lines — three-point arc, free-throw lane, centre circle — in sharp focus, "
        "the glossy floor reflecting the overhead arena lights in long stretched highlights, "
        "a hoop and backboard at the far end catching the floodlights with a white flare on the glass, "
        "player bench and scorer's table along the sideline, bleachers rising into shadow beyond",
        "high-intensity arena floodlights from above creating bright even illumination across the court, "
        "the polished wood floor acting as a massive reflector doubling the light, "
        "the bleachers falling into graduated shadow the higher they rise, "
        "scoreboards and sponsor signage providing secondary coloured light from the perimeter",
        "sneakers squeaking on the polished floor in sharp staccato bursts, "
        "the hollow bounce of a basketball on hardwood echoing through the space, "
        "crowd noise swelling from the upper tiers, a whistle blast cutting clean through the din, "
        "the metallic rattle of a ball hitting the rim and the swish of net on a clean shot"),

    # ── MEDICAL / PROFESSIONAL ──────────────────────────────────────────
    "🔬 Operating theatre — clinical white": (
        "surgical operating theatre viewed from the foot of the table, "
        "stainless steel instrument trays on wheeled stands flanking the operating table, "
        "monitors mounted on articulated arms displaying vital signs in green and amber traces, "
        "tiled floor with a central drain, walls in pale institutional green, "
        "an anaesthesia machine with its coiled tubing at the head of the table, "
        "masked and gowned figures positioned around the table in practised formation",
        "massive overhead surgical lights — multi-element LED arrays casting intense shadowless white "
        "directly onto the operating field, the light so bright it bleaches colour from the drapes, "
        "secondary ambient fluorescents providing cool fill to the rest of the room, "
        "monitor screens adding small pools of green and amber glow to the nearest surfaces",
        "the rhythmic beep of the heart monitor marking time, "
        "the steady mechanical breath of the ventilator cycling in measured intervals, "
        "quiet exchanges between surgeon and assistant — calm, clipped, technical, "
        "the clink of a metal instrument placed back on the tray"),

    "⚖ Courtroom — formal architecture": (
        "wood-panelled courtroom with dark oak wainscoting and moulding on every surface, "
        "the judge's bench elevated on a platform at the front with a high-backed leather chair, "
        "the witness stand to one side with its own small enclosure, jury box to the other "
        "with two rows of upholstered seats, counsel tables facing the bench, "
        "national and state flags flanking the bench, a court seal mounted on the wall above, "
        "public gallery rows visible at the rear behind a low wooden barrier",
        "overhead recessed fluorescents providing even institutional lighting, "
        "supplemented by brass banker's lamps on the counsel tables casting warm downward pools, "
        "tall windows on one wall with translucent shades softening daylight into flat white fill, "
        "the overall palette formal and muted — dark wood, cream walls, brass fixtures",
        "the acoustics of a formal room — every voice carrying clearly to every corner, "
        "a gavel striking with a sharp authoritative crack, papers shuffling at the counsel table, "
        "the court reporter's keys clicking in rapid quiet rhythm, "
        "the creak of wooden seating as the gallery shifts and settles"),

    # ── MUSIC ───────────────────────────────────────────────────────────
    "🎙 Recording studio — glass booth": (
        "recording studio isolation booth viewed from inside, "
        "large double-paned glass window looking out to the control room where an engineer sits at a wide mixing console, "
        "acoustic foam panels in charcoal grey covering every wall and the ceiling, "
        "a condenser microphone on a boom arm with a pop filter positioned at mouth height, "
        "a music stand with lyric sheets, closed-back headphones hanging from a wall hook, "
        "cables taped to the floor running to a junction box in the corner",
        "soft warm overhead lighting from recessed fixtures — deliberately even and non-harsh, "
        "the control room beyond the glass lit cooler with the glow of console meters "
        "and computer screens casting blue-white on the engineer's face, "
        "LED status lights on rack-mounted equipment visible through the glass in amber and green columns",
        "near-total silence inside the booth — the foam absorbing everything, "
        "the faint bleed of a click track from the headphones when no one is wearing them, "
        "the engineer's voice arriving through the talkback speaker — slightly compressed and tinny, "
        "and the particular dead acoustic quality of a room designed to have no echo at all"),

    "🎵 Concert hall — empty stage": (
        "grand concert hall viewed from the centre stalls looking toward an empty stage, "
        "soaring ceiling with ornate plasterwork and a central chandelier, "
        "tiered balconies on three sides with gilded railings and carved caryatids, "
        "rows of red velvet seats receding in gentle curves, a black Steinway grand piano "
        "alone on the stage with its lid raised, the polished surface catching the spotlight, "
        "golden ornamental details on every surface — cornices, pilasters, proscenium arch",
        "a single dramatic spotlight illuminating the piano on the otherwise dark stage, "
        "the chandelier on a low warm setting providing soft ambient fill to the auditorium, "
        "the balcony fronts catching gilt highlights from the house lights, "
        "deep shadow in the upper galleries and backstage areas beyond the proscenium",
        "the extraordinary silence of a large empty hall — not dead silence but living silence, "
        "the faintest hum of the ventilation system audible only because nothing else competes, "
        "a creak from the stage boards as temperature changes the wood, "
        "the acoustic potential of the space itself — every small sound blooming into a long natural reverb"),

    # ── WEATHER EXTREMES ────────────────────────────────────────────────
    "❄ Blizzard whiteout — visibility zero": (
        "complete whiteout conditions in a blizzard, visibility reduced to three metres, "
        "heavy snow driving horizontally on a sustained gale, "
        "dark forms — trees, fence posts, a building corner — appearing and vanishing in the white, "
        "the ground and sky indistinguishable, no horizon, no depth cues, total spatial disorientation, "
        "snow accumulating on every surface and being stripped off again by the wind",
        "no directional light — a flat featureless white-grey luminance from every direction at once, "
        "objects losing all shadow and appearing as dark silhouettes with soft undefined edges, "
        "the overall exposure blown out to near-white with only the closest dark objects registering, "
        "any artificial light source — a window, a headlamp — surrounded by a bright halo of backscattered snow",
        "wind as a wall of sound — a sustained roar with no pause and no variation in pitch, "
        "snow hitting exposed skin or fabric in a continuous abrasive hiss, "
        "all other sounds suppressed or distorted by the wind — a shout from ten feet away arriving muffled, "
        "the creak and groan of structures flexing under wind load"),

    "🌪 Tornado warning — green sky": (
        "flat rural landscape under an approaching supercell, the sky a sickly green-yellow "
        "casting unnatural light across fields and farmsteads, "
        "a visible funnel cloud descending from the wall cloud on the western horizon, "
        "debris visible at the base of the funnel — dark specks rotating upward, "
        "loose objects on the ground beginning to shift and tumble in the building wind, "
        "tree branches bending hard to one side, a screen door banging open and closed",
        "the green-tinged light from the storm clouds replacing normal daylight with an alien palette, "
        "no shadows — the overcast is total but the light has a strange intensity to it, "
        "everything lit from within by an amber-green luminance that makes colours appear wrong, "
        "distant lightning inside the cloud base illuminating the rotating structure from within",
        "wind building from steady to gusting in increasing intensity, "
        "a distant continuous roar from the tornado itself — often compared to a freight train, "
        "objects clattering and banging — a trash can rolling across pavement, metal sheeting flexing, "
        "emergency sirens wailing in long steady tones from the town behind, "
        "and sudden pressure changes making the ears pop"),

    # ── CULTURAL ────────────────────────────────────────────────────────
    "🕌 Moroccan riad — courtyard silence": (
        "interior courtyard of a traditional Moroccan riad, "
        "zellige mosaic tiles covering the lower walls in intricate geometric patterns of blue, white, and green, "
        "arched doorways on all four sides opening into shaded rooms beyond, "
        "a central stone fountain with water trickling over a mosaic basin, "
        "potted orange trees and jasmine in glazed ceramic pots flanking the fountain, "
        "carved stucco panels above the tile line, warm earth-toned plaster on the upper walls",
        "soft diffused daylight entering from the open sky directly above the courtyard, "
        "the high walls filtering and bouncing the light so it arrives even and shadowless at ground level, "
        "the tiles reflecting cool blue-green tones upward while the plaster walls return warm ochre, "
        "the shaded archways appearing dark by contrast with the bright central courtyard",
        "water from the central fountain — a continuous gentle trickling over smooth stone, "
        "birdsong from above echoing down the courtyard walls, "
        "the call to prayer arriving faintly from a distant minaret, muffled by the riad walls, "
        "a deep ambient quiet — the courtyard designed as a refuge from the medina noise outside"),

    "🛕 Indian temple — incense and sandstone": (
        "ornate sandstone temple interior, massive carved pillars rising to a ceiling "
        "covered in sculptural relief depicting mythological scenes, "
        "incense smoke rising in slow spirals from brass holders in wall alcoves, "
        "flower garlands in marigold orange and jasmine white draped over lintels and around pillars, "
        "small oil lamps — diyas — flickering in carved stone niches along the walls, "
        "the floor worn smooth by centuries of bare feet, cool stone underfoot",
        "shafts of daylight entering through narrow openings high in the walls, "
        "the light catching incense smoke and becoming visible as solid golden beams cutting the dim interior, "
        "oil lamps providing warm orange point-light sources at regular intervals along the walls, "
        "the overall palette warm sandstone gold and deep shadow with smoke-diffused edges",
        "temple bells ringing in irregular patterns — some deep and sustained, some small and bright, "
        "devotional chanting in a low resonant drone from an inner sanctum, "
        "the specific acoustic of carved stone — every sound gaining a complex reverb, "
        "incense crackling faintly as it burns, bare feet on stone, "
        "and the flutter of pigeons in the upper reaches of the ceiling"),

    "🦇 Limestone cave — stalactite cathedral": (
        "large underground limestone chamber with a cathedral-like vaulted ceiling, "
        "stalactites descending in dense clusters — some needle-thin, some massive columns "
        "where they have met stalagmites rising from the floor to form pillars, "
        "mineral deposits creating flowing curtain formations along the walls in amber and white, "
        "a still pool of water on the chamber floor reflecting the ceiling formations perfectly, "
        "the scale enormous — the far wall barely visible in the gloom",
        "a single lantern or torch providing warm directional light from one side, "
        "the light catching mineral deposits and wet surfaces in bright specular highlights, "
        "deep shadow filling the far reaches of the chamber and the spaces between formations, "
        "the water pool acting as a mirror doubling every light source from below",
        "dripping water — the primary sound — each drop a distinct note echoing through the chamber "
        "with a reverb tail lasting several seconds, "
        "the hollow resonance of the space amplifying every small sound into something immense, "
        "a faint underground stream audible from a passage leading deeper, "
        "and the total absence of wind — the air perfectly still and cool"),

    # ── SEASONAL ────────────────────────────────────────────────────────
    "🍂 Autumn forest — golden canopy": (
        "forest at peak autumn colour, the canopy a layered mosaic of amber, crimson, and burnished gold, "
        "warm light filtering through translucent leaves casting the entire forest floor in a golden wash, "
        "fallen leaves ankle-deep on the narrow path, damp and compacted beneath, dry and curling on top, "
        "mushrooms and bracket fungi growing on fallen logs and exposed roots, "
        "a spider web between two branches catching the light in a perfect geometric pattern, "
        "cool air carrying the sharp sweet smell of decomposing leaves",
        "warm diffused light filtering through the coloured canopy — amber and gold dominant, "
        "occasional shafts of direct sun breaking through gaps to create bright pools on the forest floor, "
        "the overall palette saturated warm — every surface tinted by the canopy filter, "
        "long soft shadows from the tree trunks with warm-toned fill from reflected leaf light",
        "leaves falling in slow irregular spirals, each one audible as a faint tick on landing, "
        "a breeze moving through the canopy creating a dry rustling like distant applause, "
        "a woodpecker drumming on a dead trunk in measured bursts, "
        "the crunch of leaves underfoot and the particular silence of a forest in autumn — "
        "fewer birds, fewer insects, the year winding down"),

    "🏔 Winter cabin — firelight": (
        "log cabin interior in deep winter, rough-hewn timber walls with white chinking between the logs, "
        "a large stone fireplace dominating one wall with a substantial fire burning in it, "
        "heavy wool blankets and fur throws draped over a deep armchair and a wooden bench, "
        "frost patterns crystallised on the inside of small-paned windows, "
        "snow visible outside through the glass — deep, blue-white, and still, "
        "a kettle on an iron hook near the fire, bookshelves built into the wall beside the chimney",
        "the fireplace as the primary light source — warm flickering orange filling the room unevenly, "
        "shadows dancing on the ceiling and walls with every shift of the flames, "
        "the windows admitting a cold blue-white ambient from the snow outside "
        "creating a strong warm-cool contrast between the fire side and the window side of the room",
        "the fire — crackling, popping, the occasional hiss of sap in a log, "
        "a settling shift as a log breaks and drops in the grate, "
        "wind outside audible as a low moan around the eaves and chimney, "
        "the tick of contracting wood in the cabin frame as the temperature drops, "
        "and deep snow silence pressing in from every direction outside the walls"),
}


# ═══════��═════════════════════════════════════��════════════════════════════
#  ANIMATION PRESETS
#  Pre-loaded character universes for cartoons natively trained in LTX 2.3
# ═══���══════════════════════════════════════════════════════════════════════
ANIMATION_PRESETS = {
    "None": None,

    "SpongeBob SquarePants": {
        "style_tag": "SpongeBob SquarePants animation, Nickelodeon 2D cartoon style, vibrant underwater colours, exaggerated expressions",
        "characters": {
            "SpongeBob": "yellow square sponge, huge blue eyes, buck teeth, red tie, brown square pants, optimistic and energetic",
            "Patrick": "pink starfish, green floral swim trunks, vacant expression, lovable but dim",
            "Squidward": "blue-green octopus, long drooping nose, four tentacle legs, perpetually annoyed, cashier shirt and brown pants",
            "Mr. Krabs": "red crab, big pincer claws, tiny eyestalks, business shirt, money-obsessed, gravelly voice",
            "Sandy": "squirrel in white diving suit with clear dome helmet, air hose, flower decal, Texan accent, scientist",
            "Plankton": "microscopic green copepod, single eyestalk, villain, obsessed with Krabby Patty formula, shrill voice",
        },
        "locations": [
            "Krusty Krab interior — ship-shaped restaurant, order counter with cash register, grill station visible through kitchen window, wooden booths and tables, porthole windows, Mr. Krabs office door with dollar sign, squeaky floorboards",
            "SpongeBob pineapple house — living room with Gary's snail tank, coral furniture, porthole windows, kitchen with pineapple appliances, spiral staircase, framed jellyfishing prints",
            "Jellyfish Fields — vast rolling underwater meadows, clouds of pink jellyfish drifting in slow patterns, soft dappled light from the ocean surface above, coral outcroppings with nets leaning against them",
            "Bikini Bottom streets — coral-built storefronts along curved road, anchors and ship wheels as signage, bubble transitions between scenes, sea creatures in cars, Krusty Krab visible on the hill",
            "Squidward tiki house — moody dark interior directly between SpongeBob and Patrick houses, easel with self-portrait, clarinet on stand, reading chair, windows uncomfortably close to SpongeBob pineapple",
            "Sandy treedome — giant glass air-sealed dome on ocean floor, Texas ecosystem inside: oak tree, flower beds, rope swing, science equipment, airlock entrance requiring water helmet",
            "The Chum Bucket — dingy grey exterior across from Krusty Krab, computer wife Karen on wall inside, Plankton laboratory below, perpetually empty of customers, world domination blueprints on walls",
        ],
        "tone": "high-energy slapstick, nautical puns, exaggerated physical comedy, optimistic chaos, underwater absurdism",
    },

    "Bluey": {
        "style_tag": "Bluey animation, BBC Studios Australian cartoon style, soft pastel colours, simple expressive characters, warm domestic lighting",
        "characters": {
            "Bluey": "blue heeler puppy, 6 years old, imaginative and energetic leader, blue fur",
            "Bingo": "red heeler puppy, 4 years old, sweet and earnest younger sister, red-orange fur",
            "Bandit": "blue heeler dad, patient and playful, gets roped into imaginative games, wears casual clothes",
            "Chilli": "red heeler mum, warm and grounded, occasionally exasperated, works part-time",
        },
        "locations": [
            "Heeler backyard — timber deck with outdoor furniture, Hills Hoist clothesline, trampoline with safety net, large gum tree, Brisbane suburban garden with patchy grass, back fence to neighbour yard, afternoon golden light through leaves",
            "Heeler living room and kitchen — open plan, low couch with cushions, coffee table with toys, TV on wall, kitchen island behind, crayon drawings on fridge, school bags near door, warm interior light",
            "Heeler kids bedroom — bunk beds Bluey on top Bingo below, toy shelves, soft toys scattered, glow-in-dark stars on ceiling, Bluey drawings pinned to wall, nightlight on bedside table",
            "School playground — colourful climbing equipment, bark chip ground, shade sails overhead, bench where parents wait, Brisbane suburban primary school feel, friends Chloe and Judo and Mackenzie",
            "Creek and bushland — rocky creek bed with shallow water, gum trees overhead, wattles in flower, birds in canopy, kids catching tadpoles in jam jars, dappled Australian bush light",
            "Swim school — indoor pool with floating lane dividers, echoing acoustics, swimming instructor, changing rooms corridor, chaos of dog children learning to swim",
            "Dad work office — open plan architecture office, big desks with drawings pinned up, Bandit colleagues, the game that takes over the whole office when Bluey visits",
        ],
        "tone": "gentle heartwarming, imaginative play sequences, emotional honesty for children and adults, soft Australian humour, games with loose rules",
    },

    "Peppa Pig": {
        "style_tag": "Peppa Pig animation, simple 2D British cartoon style, flat colour backgrounds, minimal detail, bright primary colours",
        "characters": {
            "Peppa": "pink pig, round body, simple design, confident and slightly bossy, red dress",
            "George": "smaller pink pig, loves dinosaurs, says Dine-saw",
            "Mummy Pig": "pink pig, patient and gentle, works on computer",
            "Daddy Pig": "larger pink pig, round belly, cheerful and clumsy, loves his car",
            "Grandpa Pig": "older pink pig, captain hat, owns a boat and vegetable garden",
            "Granny Pig": "older female pig, kind, makes cakes",
            "Suzy Sheep": "white sheep, Peppa best friend, competitive, pink dress",
        },
        "locations": [
            "Peppa house — simple two-storey on a hill, round windows, front door facing garden, muddy puddle directly outside front gate, Daddy Pig car in drive, simple green garden, flat horizon behind",
            "The muddy puddle — the most important location in the show, outside front gate, brown and always inviting, entire family jumps in it at episode end, Wellington boots mandatory",
            "Grandpa Pig house — slightly larger, vegetable patch with carrots and cabbages, pond, shed full of tools, small greenhouse, vegetable garden as episode source",
            "Grandpa Pig boat — small vessel in harbour or canal, below deck cabin, rope and anchor, the boat that always needs fixing, seaside setting with seagulls",
            "Playgroup — single-room classroom, small tables and chairs, Madame Gazelle at front with guitar, paintings drying on line, dressing up corner",
            "Public swimming pool — changing rooms, the big pool and the small pool, Daddy Pig jumping in with enormous splash",
            "Daddy Pig office — open plan with computers, pig colleagues, his important spreadsheets, the photocopier",
        ],
        "tone": "simple gentle British politeness, muddy puddles are the highest joy, family dynamics played straight, everyone laughs at the end",
    },

    "Looney Tunes (Classic)": {
        "style_tag": "Looney Tunes classic animation, Warner Bros 1940s-60s 2D cartoon style, painted backgrounds, fluid anarchic movement",
        "characters": {
            "Bugs Bunny": "grey rabbit, white gloves, casual confidence, Brooklyn accent, always one step ahead, What is up Doc",
            "Daffy Duck": "black duck, white ring around neck, lisp, easily jealous, You are despicable",
            "Elmer Fudd": "rotund hunter, red jacket, hunting rifle, speech impediment turning R and L to W, hunting Bugs",
            "Tweety": "small yellow canary, large head, innocent face, surprisingly resourceful, I tawt I taw a puddy tat",
            "Sylvester": "black and white tuxedo cat, perpetually chasing Tweety, Sufferin succotash",
            "Wile E. Coyote": "grey coyote, obsessed with catching Road Runner, uses ACME products, always fails",
            "Road Runner": "blue-purple bird, Beep Beep, always escapes, impossibly fast",
            "Yosemite Sam": "tiny man, enormous red moustache, twin pistols, hair-trigger temper",
        },
        "locations": [
            "American southwestern desert — Monument Valley red rock formations, single road to horizon, cactus, painted cliff tunnel that only Road Runner can pass through, ACME delivery addresses on rocks, canyon edges extending further than possible",
            "Elmer Fudd hunting forest — dense painted woodland, rabbit holes bigger on inside, hunter cabin with antler trophies, seasonal changes mid-episode, the rabbit season duck season sign in the clearing",
            "Granny house — Victorian townhouse, Tweety cage in bay window, Granny umbrella by door, rocking chair, basement where Sylvester ends up, back garden with bulldog kennel",
            "City street — Warner Bros backdrop urban setting, manholes characters disappear into, buildings that collapse in cartoon physics ways, the fire hydrant that always gets opened",
            "Opera house — for Carl Stalling orchestra pieces, stage and pit, seats full of animal audience, the conductor whose score characters disrupt",
        ],
        "tone": "anarchic slapstick, physics only apply when convenient, ACME products always fail, character survives anything, Warner Bros orchestral musical timing",
    },

    "Toy Story / Pixar": {
        "style_tag": "Toy Story Pixar CGI animation style, warm detailed environments, toys with expressive plastic faces, photorealistic lighting on toy surfaces",
        "characters": {
            "Woody": "cowboy doll, pull-string on back with voice box, plaid shirt, cowboy hat, loyal leader, anxious when threatened",
            "Buzz Lightyear": "space ranger action figure, purple and white, wing buttons, wrist communicator, originally deluded about being real",
            "Jessie": "cowgirl doll, red hat, braid, energetic, yodels, abandonment trauma",
            "Rex": "green plastic T-rex, anxious, tiny arms, large roar he is proud of",
            "Hamm": "pink piggy bank, coin slot on back, sarcastic, carries the change",
            "Mr. Potato Head": "plastic potato body, detachable facial features, sarcastic, Brooklyn attitude",
            "Slinky Dog": "coiled spring body, front and back dog halves, loyal, stretches to bridge gaps",
        },
        "locations": [
            "Andy bedroom — single bed with cowboy bedspread, toy box under window, bookshelf, Woody roundup poster on wall, model rocket on desk, window to suburban street, afternoon light casting long toy shadows, toys arranged where Andy left them",
            "Andy living room — carpet where toys walk, sofa toys hide under, TV and VCR, the stairs as major obstacle, the baby monitor that overhears conversations",
            "Pizza Planet — 1990s American space-themed restaurant, rocket ship in car park, arcade machines, UFO claw machine full of alien squeeze toys who worship the claw, neon lighting, sticky carpets",
            "Sid bedroom — dark curtains drawn, dismantled toy parts everywhere, tool bench with half-finished experiments, black walls with skull stickers, broken toys living under the bed and in shadows",
            "Al toy barn and apartment — museum-quality display cases, mint-in-box collectors items, Japanese collectors waiting by fax machine, the Woody Roundup VHS tapes playing on TV",
            "Sunnyside Daycare — bright colourful room that looks welcoming, Lotso territory, toddler room with chaos, older kids room with structure, the dumpster outside as final threat",
            "Bonnie bedroom — smaller and warmer than Andy, handmade toys alongside commercial ones, drawings on wall, child who plays differently and more imaginatively",
        ],
        "tone": "emotional depth beneath toy comedy, toys have loyalty and anxiety, freeze instantly when humans appear, friendship and belonging themes",
    },

    "Batman (LEGO)": {
        "style_tag": "LEGO Batman animation style, CGI brick-built world, everything made of LEGO including explosions and water, bright primary colours, visible stud textures on all surfaces",
        "characters": {
            "Batman": "LEGO minifigure in black bat suit, cowl with pointed ears, utility belt with LEGO pouches, self-serious, secretly lonely, I work alone, plays Nine Inch Nails in the Batmobile",
            "Robin": "yellow cape, red and green suit, bowl cut hair piece, eager sidekick, calls Batman by name never dad though he wants to",
            "The Joker": "green hair piece, purple LEGO suit, wide printed smile, wants Batman to acknowledge him as greatest enemy, genuinely hurt when Batman denies their relationship",
            "Alfred": "butler minifigure, white hair, black jacket, patient, delivers emotional wisdom as dry wit, concerned about Batman emotional health",
            "Barbara Gordon": "red hair, purple police uniform becoming Batgirl suit, competent, immediately better at Batman job than Batman",
        },
        "locations": [
            "The Batcave — enormous underground LEGO space, giant penny on wall built from bricks, LEGO dinosaur skeleton, Bat-computer with multiple screens, Batmobile on platform, suits on display pedestals, brick-built stalactites, Alfred serving tea at bottom of main stairs",
            "Wayne Manor — grand LEGO mansion on cliff, enormous ballroom, portrait gallery of Wayne ancestors, hidden cave entrance below, Alfred quarters, Bruce enormous empty bedroom with robot dancing equipment",
            "Gotham City streets — all-brick cityscape at night, LEGO cars and buses, brick-built rain falling as flat pieces, rogues gallery hideouts across skyline, Arkham on the hill, police station on corner",
            "Arkham Asylum — LEGO brick prison, comically poor security, rotating villain population, warden office, common room where villains socialise between escapes",
            "The Phantom Zone — flat black and white brick space, flat 2D brick versions of criminals imprisoned there, weird geometry, the projector that opens and closes it",
        ],
        "tone": "self-aware superhero parody, Batman ego is the joke, emotional growth hidden under action comedy, everything is awesome",
    },

    "Scooby-Doo": {
        "style_tag": "Scooby-Doo animation, Hanna-Barbera 2D cartoon style, limited animation with held poses, painted atmospheric mystery location backgrounds",
        "characters": {
            "Scooby-Doo": "large brown Great Dane, SD collar tag, speaks broken English adding R sounds, cowardly but brave when Scooby Snacks are offered, Scooby-Dooby-Doo",
            "Shaggy": "lanky teenager, green shirt, brown bell-bottoms, scraggly chin, always hungry, best friend with Scooby, Zoinks",
            "Velma": "short, orange roll-neck sweater, thick square glasses she loses at worst moments, smartest in group, Jinkies",
            "Daphne": "red hair, purple dress and headband, scarf, danger-prone Daphne, more capable than people assume",
            "Fred": "blond, white shirt with orange neckerchief, trap-builder who overestimates his traps, team leader",
        },
        "locations": [
            "Haunted mansion — Victorian exterior with rusted gates, cobwebs on every surface, grand entrance hall staircase, secret passages behind bookcases, flickering candelabras, portrait eyes that follow the gang, basement boiler room, attic with covered furniture",
            "Mystery Machine van — painted green van with flower, front seats for Fred and Daphne, back area for others, maps and equipment, Scooby snacks in glove box",
            "Spooky graveyard — cast iron fence, fog at knee height, tilted headstones, bare trees, mausoleum in centre, moonlight as only source, groundskeeper hut at edge",
            "Abandoned amusement park — rusted Ferris wheel still slowly turning, funhouse with distorting mirrors, dark ride tunnel, cotton candy cart tipped over, padlocked main gate with Closed sign",
            "Old lighthouse — coastal cliff, light mechanism still working, spiral stairs, fog horn, rocks below, smuggler cave accessible at low tide, keeper quarters with logbook",
            "Old theatre or opera house — velvet seats with springs, stage with rigging, dressing rooms, orchestra pit, flies above stage full of dropped scenery",
        ],
        "tone": "mystery comedy formula, monster always a person in a mask with property motive, Scooby Snacks bribe, chase sequence with musical cue, mask reveal ending, gang splits up despite knowing it is a bad idea",
    },

    "He-Man": {
        "style_tag": "He-Man Masters of the Universe animation, 1980s Filmation cartoon style, limited animation with static holds, bold heroic character designs, vivid primary colours",
        "characters": {
            "He-Man": "enormously muscular blond hero, fur loincloth and harness, Power Sword glowing, By the power of Grayskull transformation sequence, speaks in declarative heroic sentences",
            "Skeletor": "blue humanoid skin, yellow bare skull face, purple hood and body armour, havoc staff with ram skull top, high-pitched nasal evil laughter, surrounded by incompetent minions",
            "Battle Cat": "enormous green tiger with yellow saddle and armour, He-Man mount, Cringer when not transformed",
            "Man-At-Arms": "brown and orange armour with distinctive moustache, royal engineer and weapons master, builds the vehicles",
            "Teela": "white armour, auburn hair, warrior goddess captain of royal guard, independent and fierce",
            "Orko": "small floating magician, red hat, scarf covering face, magic that always goes wrong for comic relief",
            "Evil-Lyn": "Skeletor second in command, yellow skin, dark sorceress, more competent than Skeletor",
        },
        "locations": [
            "Castle Grayskull — enormous skull-shaped fortress rising from bottomless chasm, jawbridge entrance that lowers like a jaw, Sorceress throne inside, ancient power radiating from walls, surrounding rock formations and eternal mist",
            "Royal Palace of Eternia — white and gold towers against blue sky, throne room with King Randor and Queen Marlena, training courtyard, Man-At-Arms workshop below, rooftop overlooking Eternia city, royal guards in formation",
            "Snake Mountain — Skeletor dark fortress shaped like giant serpent head, scaly rock exterior, throne room inside the mouth, dungeon below, Evil-Lyn tower with crystal ball, surrounding toxic landscape of jagged rocks",
            "Eternia landscape — alien terrain combining jungle desert and crystal formations, twin moons in purple sky, the road between palace and Castle Grayskull, ancient ruins of previous civilisations",
            "The Fright Zone — evil dimension controlled by Evil Horde, swamp and decay, Hordak fortress, weeping willows that scream",
        ],
        "tone": "heroic 1980s moral clarity, good vs evil with no ambiguity, inspirational closing message direct to camera, power fantasy with honour, He-Man never kills always finds non-lethal solution",
    },

    "Shrek": {
        "style_tag": "Shrek DreamWorks CGI animation style, fairy tale world with subversive edge, detailed medieval environments, highly expressive faces, early 2000s CGI with impressive natural detail",
        "characters": {
            "Shrek": "large green ogre, Scottish accent, ears like suction cups, I am like an onion I have layers, reluctant hero who wants to be left alone in his swamp",
            "Donkey": "grey donkey, Eddie Murphy energy, over-shares everything, desperately wants to be Shrek friend, has a dragon wife now",
            "Fiona": "red hair tied back, green dress, secretly an ogre at night, fierce and capable, rescues herself before Shrek arrives",
            "Puss in Boots": "orange tabby cat, Spanish accent, musketeer hat, tiny boots, enormous persuasive eyes as weapon, sword fighter",
            "Lord Farquaad": "very short man, black bowl cut, square jaw, ruler of spotless Duloc, compensating for height through architecture and cruelty",
            "Dragon": "enormous red dragon, female, married Donkey, breathes fire, surprisingly gentle when not threatened",
        },
        "locations": [
            "Shrek swamp — muddy pool with handmade KEEP OUT signs, wooden outhouse, sunflower garden, rustic one-room interior with mud bath, candles made from earwax, the specific solitude Shrek constructed around himself",
            "Far Far Away — fairy tale kingdom styled after Beverly Hills, enormous castle on hill, main street with Farbucks Coffee and Fiona face on every billboard, fairytale creatures Farquaad expelled living on outskirts",
            "Duloc — sterile white and gold medieval theme park city, perfectly geometric squares of grass, the welcome song in the information booth, Farquaad enormous castle relative to tiny citizens",
            "Dragon castle — crumbling medieval fortress on volcanic island, lava moat, drawbridge, Dragon lair inside with hoard, partially collapsed bridge",
            "Fairy Godmother factory — industrial magical production facility, conveyor belts of potions, workers in pointed hats, piano for her cabaret number, vast catalogue of happy endings for purchase",
        ],
        "tone": "subversive fairy tale, beauty and outsider themes, crude humour alongside genuine emotion, fairy tale conventions inverted and sometimes restored, the swamp as paradise",
    },

    "Madagascar (Lemurs)": {
        "style_tag": "Madagascar DreamWorks CGI animation style, bright tropical jungle setting, highly expressive cartoon animal characters, warm saturated tropical colour palette",
        "characters": {
            "King Julien": "ring-tailed lemur, golden crown, red cape, absolute monarch of questionable legitimacy, I like to move it, dance-obsessed, oblivious to danger",
            "Maurice": "aye-aye lemur, large eyes, King Julien long-suffering advisor, only one who sees problems coming, perpetually worried",
            "Mort": "tiny mouse lemur, enormous innocent eyes, obsessed with touching King Julien feet, childlike, surprisingly resilient",
            "Alex": "lion from New York, mane styled like celebrity, loves performing and being adored, out of his depth in actual wild",
            "Marty": "zebra from New York, wants to see the wild, philosophical about his purpose",
            "Gloria": "hippo, pragmatic and warm, surprisingly graceful in water, strongest member of the group",
            "Melman": "giraffe, hypochondriac, actually a doctor now, tallest vantage point",
        },
        "locations": [
            "Lemur kingdom Madagascar jungle — King Julien throne atop giant baobab tree, lemur village in canopy below with huts, dance floor clearing with torches, the sacrificial volcano at territory edge that Julien makes offerings to",
            "Madagascar beach — long white sand beach where New York zoo animals washed up, arrival crates still on sand, jungle rising immediately behind, lagoon for swimming, logs used as furniture",
            "New York Central Park Zoo — spacious enclosures, penguin habitat at corner, Alex performing enclosure, visitors behind the rail, the famous Alex the Lion sign",
            "African savanna — the actual wild Marty imagined, watering hole, wide open grass, reality versus fantasy of nature documentaries",
            "Penguin submarine — military interior, sonar equipment, periscope, the penguins vessel for all their operations",
        ],
        "tone": "King Julien deluded royalty as primary comedy engine, Mort innocent obsession, jungle as absurd paradise where being dangerous is a social problem, city animals confronting nature",
    },

    "Despicable Me (Minions)": {
        "style_tag": "Despicable Me Illumination CGI animation style, yellow Minion designs, warm villain-lair palette, suburban neighbourhood contrast, smooth rounded character surfaces",
        "characters": {
            "Gru": "tall grey villain, Eastern European accent, enormous pointed nose, bald head, black coat, reformed villain navigating fatherhood, loves his daughters",
            "Minion generic": "yellow pill-shaped creature, blue overalls with Gru logo, one or two circular goggle eyes, speaks Minionese mixing English Spanish French Italian with banana sounds, simple desires: banana, music, chaos",
            "Kevin": "tall two-eyed Minion, slightly more capable than average, self-appointed leader",
            "Stuart": "medium one-eyed Minion, plays guitar, easily distracted by shiny things and food",
            "Bob": "small round two-eyed Minion with one brown eye one green eye, carries stuffed bear named Tim, childlike innocence",
            "Dr. Nefario": "elderly villain scientist, thick glasses, lab coat, mishears instructions catastrophically, built fart gun when Gru asked for dart gun",
        },
        "locations": [
            "Gru underground lair — beneath the suburban house, enormous underground facility, Minion dormitories in bunk beds stretching into distance, rocket hangar, weapon development laboratory, big pink plotting chair, liquid hot magma chamber, jelly gun testing range",
            "Gru suburban house — dark gothic house on otherwise normal street, neighbours who complain, kitchen where Gru serves girls breakfast, living room that gets destroyed regularly",
            "Vector pyramid fortress — modern high-tech villain base near ocean, luxury interior, shark tank, shrink ray storage, unnecessarily complicated security Vector is proud of",
            "Bank of Evil — formerly Lehman Brothers, where villains apply for loans, waiting room of villains reading Evil Weekly, the loans officer who evaluates evil plans",
            "Villain-Con — annual convention of supervillains, booths selling weapons and evil plans, awards ceremony, villain social hierarchy on display",
        ],
        "tone": "Minion chaos as primary visual comedy, banana obsession and Minionese gibberish, fart guns and shrink rays, villain redemption through unexpected parenthood, Minions simple worldview as emotional core",
    },

    "Avatar: The Last Airbender": {
        "style_tag": "Avatar The Last Airbender Nickelodeon animation style, anime-influenced 2D with fluid bending action sequences, rich elemental visual effects, detailed world-building across four nations",
        "characters": {
            "Aang": "young Air Nomad, completely bald with blue arrow tattoos on forehead and hands, orange and yellow monk robes, airbending staff, playful and compassionate, carries weight of being last Avatar",
            "Katara": "Water Tribe girl, brown skin, dark hair in characteristic loops, blue Water Tribe clothing, waterbending master, maternal and determined, healer",
            "Sokka": "Water Tribe warrior, brown skin, dark hair in wolf-tail, blue outfit, boomerang and space sword, non-bender who compensates with tactics and humour",
            "Toph": "blind earthbender, bare feet always on ground to sense vibrations, green Earth Kingdom clothing, tough sarcastic exterior, genuinely the most powerful bender in the group",
            "Zuko": "Fire Nation prince, scar covering left side of face from his father, top-knot then free hair during redemption arc, conflicted honour, firebending",
            "Uncle Iroh": "heavyset retired Fire Nation general, top-knot, tea-obsessed, wise beneath humble surface, genuine warmth, the Dragon of the West",
            "Azula": "Fire Nation princess, dark hair, blue fire instead of orange, ruthless perfectionist, psychological warfare as primary weapon",
        },
        "locations": [
            "Southern Air Temple — high mountain peak, circular architecture with open arches and wind channels, sky bison stables carved from peak, meditation platforms, the sanctuary with past Avatar statues, Pai Sho table, now abandoned and windswept",
            "Fire Nation palace — imperial red and black architecture on volcano island, throne room with wall of fire Ozai speaks through, war room table for strategic planning, palace gardens, Fire Lord private chambers",
            "Southern Water Tribe — ice architecture, circular village plan around central meeting space, spirit water healing pool, wolf-otter pens, longboats on ice, aurora australis overhead at night",
            "Ba Sing Se — enormous walled Earth Kingdom city, multiple concentric rings with different social classes, the Upper Ring with palace and wealthy, Lower Ring with workers, monorail connecting rings, Long Feng Dai Li headquarters underground",
            "Western Air Temple — built into underside of cliff face, architecture that hangs upside down, perfect gaang refuge, waterfalls nearby, abandoned kitchens and dormitories",
            "Ember Island — Fire Nation holiday resort, beach house of royal family, Ember Island Players theatre, the moment of relaxation before Sozin comet",
            "The Spirit World — parallel dimension accessed through meditation, twisted landscape where emotions become environment, Wan Shi Tong library, Koh the Face Stealer lair, no rules of physics apply",
        ],
        "tone": "war and colonisation themes with genuine nuance, found family dynamics built slowly, honour and redemption arcs with real cost, elemental philosophy as character philosophy, genuine emotional stakes",
    },

    "BoJack Horseman": {
        "style_tag": "BoJack Horseman Netflix animation style, half-human half-animal anthropomorphic characters, detailed Los Angeles backgrounds, painterly colour palette, dark comedy aesthetic",
        "characters": {
            "BoJack Horseman": "anthropomorphic horse, brown fur, dark mane, blue sweater with yellow stars, tall and broad, 90s sitcom has-been, self-destructive, sardonic, genuinely funny but deeply sad",
            "Princess Carolyn": "anthropomorphic pink cat, always in business attire, sharp bob haircut, high heels, relentlessly driven agent/manager, competent and guarded",
            "Todd Chavez": "human man, early 20s, messy dark hair, red hoodie, lives on BoJack's couch, absurdist schemes, earnest and chaotic good",
            "Diane Nguyen": "human Vietnamese-American woman, glasses, dark hair, writer and activist, thoughtful and anxious, perpetually disillusioned",
            "Mr. Peanutbutter": "anthropomorphic yellow Labrador, always smiling, boundless enthusiasm, 90s sitcom rival to BoJack, genuinely kind but oblivious",
        },
        "locations": [
            "BoJack's Hollywood Hills mansion — mid-century modern, pool, panoramic city view, always slightly messy, empty alcohol bottles",
            "Hollywoo — Los Angeles where the D fell off the Hollywood sign, anthropomorphic animals and humans coexist on the streets, film industry everywhere",
            "Princess Carolyn's agency office — sleek, glass walls, industry awards, frantic energy",
            "A shot bar or restaurant — BoJack drinking alone or with reluctant company",
            "The set of Horsin' Around — 90s sitcom set, studio lights, live audience, BoJack in his element and out of time",
        ],
        "tone": "dark comedy masking genuine tragedy, addiction and depression treated honestly, anthropomorphic animal visual gags alongside real emotional devastation, Hollywood satire, characters trying and failing to be better people",
    },

    "Rick and Morty": {
        "style_tag": "Rick and Morty Adult Swim animation style, crude 2D line work with detailed grotesque alien designs, body horror transformations, interdimensional neon colour palettes, deliberately inconsistent proportions",
        "characters": {
            "Rick": "spiky light-blue hair, white lab coat always stained, flask in hand or pocket, burping mid-sentence mid-word, thin string of drool on chin, nihilistic genius who genuinely loves his family despite everything",
            "Morty": "yellow polo shirt tucked into blue jeans, unibrow, slightly hunched anxious posture, stammering speech pattern I-I-I mean, genuinely good heart being slowly corrupted, rare moments of actual confidence",
            "Beth": "blonde hair, hospital scrubs or casual clothes, horse heart surgeon, wine glass almost always present, caught between being her father daughter and being a good mother",
            "Jerry": "meek ineffectual dad, khaki slacks and polo, genuinely loves his family and is genuinely bad at most things, occasionally surprisingly competent",
            "Summer": "teenage girl, red hair, phone nearly always in hand, more competent and morally flexible than she first appears, absorbed more of Rick worldview than Rick intended",
        },
        "locations": [
            "Rick garage — the real headquarters of everything, portal gun hanging on wall, Rick ship folded into small cube on workbench, alien tech in various states of assembly, fluid-stained floor, the mini-Eiffel Tower Rick built for no reason, garage door opening to suburban driveway",
            "Smith family living room — suburban American couch and flatscreen TV, Beth horse paintings, Jerry failed home improvement attempts, the TV they watch intergalactic cable on, completely normal until Rick comes through",
            "Rick ship interior — surprisingly spacious, pilot seat, navigation AI, toilet that is also a portal, weapons systems used casually",
            "Alien planet — each one completely different: gassy atmospheres with floating rock platforms, ocean worlds where everything is sea creature, hivemind planets, Medieval planets with dragons that are actually spaceships, the specificity is the joke",
            "Citadel of Ricks — interdimensional space station city populated entirely by alternate versions of Rick: background characters are ALL Rick variants in different outfits and styles (cowboy Rick, ninja Rick, business Rick, punk Rick, fat Rick, cop Rick), with Morty variants as the underclass doing service jobs, futuristic architecture, Rick currency, bureaucratic signage, presidential podium visible in distance",
            "Blips and Chitz arcade — intergalactic arcade, Roy A Life Well Lived simulation pod, the tickets and prize counter, alien bar next door",
            "Interdimensional customs — bureaucratic portal authority, space between dimensions with its own geography, the Council of Ricks former headquarters ruins",
        ],
        "tone": "dark sci-fi comedy, existential nihilism with emotional undercurrent that breaks through unexpectedly, rapid-fire dialogue rewarding attention, gross-out body horror as casual background detail, the show aware of its own cynicism",
    },

    "Studio Ghibli": {
        "style_tag": "Studio Ghibli animation, Miyazaki hand-painted dreamlike style, watercolour textures with visible brushstrokes, lush pastoral and magical settings, soft diffused lighting",
        "characters": {
            "Chihiro": "ten-year-old Japanese girl, short brown hair with a purple hair tie, white and green striped shirt, red shorts, determined wide eyes, small frame moving through a world too large for her, spirit world worker uniform later: white top and baggy pink pants",
            "Totoro": "enormous grey forest spirit, round soft belly, pointed ears, wide toothy grin, tiny nose, whiskers, leaf on head in the rain, can only be seen by children, silent except for a deep rumbling roar",
            "Howl": "tall young wizard, sharp pale face, bright blue eyes, flowing blond hair that changes to black when distressed, jewelled earrings, extravagant jacket over loose shirt, vain and dramatic but genuinely kind",
            "Sophie": "young woman aged into an elderly body by a curse, silver hair in a bun, plain blue dress and apron, hunched posture that straightens as confidence grows, warm knowing eyes regardless of her apparent age",
            "San": "feral wolf-girl, dark hair, red triangular face paint, white fur cape and hood, crystal dagger, fierce scowl, raised by wolf gods, moves on all fours when running with her wolf siblings",
            "Kiki": "thirteen-year-old witch, large red bow in short dark hair, dark purple dress, broomstick always nearby, black cat Jiji on her shoulder, earnest and independent, learning to fly on her own",
        },
        "locations": [
            "Bathhouse — enormous traditional Japanese bathhouse rising from still water, red and gold lacquered wood, hundreds of paper lanterns glowing at night, steaming open-air pools, wooden walkways over dark water, spirit guests of every shape drifting through corridors, Yubaba office at the very top with enormous ornate doors",
            "Forest Spirit realm — ancient cedar forest with moss covering every surface, dappled green light filtering through canopy, kodama tree spirits clicking their heads in the branches, a still sacred pool reflecting the sky, enormous roots forming natural archways, the air thick with pollen and floating seeds",
            "Moving Castle — impossible mechanical assemblage of turrets chimneys and legs, clanking and hissing steam as it walks across flower-covered hills, interior larger than exterior with cluttered wizard workshop, Calcifer the fire demon in the hearth, a door that opens to four different places depending on the dial",
            "Laputa floating city — ancient overgrown sky fortress above the clouds, crumbling stone walls overtaken by enormous tree roots, a central crystal powering everything, robot guardians covered in moss tending gardens, wind blowing through empty corridors, clouds drifting through broken windows",
            "Kiki bakery — small European seaside bakery with a bell above the door, flour-dusted counter, warm bread smell, Kiki delivery broomstick leaning by the entrance, upstairs attic bedroom with ocean view, cobblestone street outside with bicycles and flower boxes",
        ],
        "tone": "gentle wonder, environmental reverence, coming-of-age journeys, hand-painted warmth in every frame, quiet moments given as much weight as dramatic ones, nature as living presence",
    },

    "Dragon Ball Z": {
        "style_tag": "Dragon Ball Z animation, 90s Toei anime action style, bold black outlines, high saturation colours, speed lines radiating from impacts, dramatic power-up auras, screaming transformation sequences",
        "characters": {
            "Goku": "muscular Saiyan warrior, wild spiky black hair that turns golden when Super Saiyan, orange gi with blue undershirt and belt, blue wristbands and boots, cheerful innocent face that hardens in battle, tail scar on lower back",
            "Vegeta": "shorter but powerfully built Saiyan prince, sharp widow's peak black hair standing straight up, blue bodysuit under white Saiyan armour with gold trim, perpetual scowl, white-gloved fists clenched, royal arrogance in every gesture",
            "Frieza": "small pale alien tyrant, purple and white bio-armour skin, long tail, red eyes, black lips curled into a smirk, hovering in his pod or standing with hands clasped behind his back, multiple transformation forms each more terrifying",
            "Gohan": "Goku young son, bowl-cut black hair, purple gi or Piccolo-style training outfit, enormous hidden power that erupts under emotional duress, gentle scholar personality that hides the strongest fighter",
            "Piccolo": "tall green Namekian, pointed ears, antennae, pink muscle patches on arms, white turban and weighted cape he removes before serious fights, stoic mentor figure, arms crossed, meditating while floating",
            "Cell": "green and black bio-android, segmented exoskeleton, spotted pattern, wings in perfect form, long tail with stinger, arrogant smirk, black pupils in pink eyes, combines traits of all fighters",
        },
        "locations": [
            "World Tournament arena — raised stone fighting platform in open stadium, enormous crowd in circular stands, tournament announcer at ringside, cracked tiles from previous impacts, blue sky above, fighter waiting area behind stone walls",
            "Planet Namek — alien world with green sky and blue-green grass, three suns casting multiple shadows, Namekian villages of round white buildings, dragon balls scattered across the landscape, enormous craters from battles scarring the terrain",
            "Hyperbolic Time Chamber — infinite white void with a small living quarters at the entrance platform, time flows differently inside, featureless horizon in every direction, extreme gravity and temperature shifts, a year inside equals one day outside",
            "West City — futuristic metropolis with capsule-shaped buildings, flying cars, Capsule Corporation dome headquarters with the company logo, Dr. Briefs laboratory inside, Vegeta gravity training room glowing red through its windows",
            "Kami Lookout — circular platform floating high above Earth, white tiled floor, palm trees in ornamental pots, entrance to the Hyperbolic Time Chamber, Mr. Popo tending the gardens, panoramic view of the world curving below through clouds",
        ],
        "tone": "escalating power levels, dramatic transformation sequences with screaming and lightning, friendship and rivalry driving every fight, five-minute battles lasting ten episodes, the spirit bomb always takes too long to charge",
    },

    "Naruto": {
        "style_tag": "Naruto anime animation, Studio Pierrot style, dynamic ninja combat, hand-sign jutsu sequences, forehead protector insignias, bold action lines, leaf motifs",
        "characters": {
            "Naruto": "blond spiky hair, bright blue eyes, orange and black jumpsuit with red spiral on the back, whisker marks on both cheeks from the Nine-Tailed Fox sealed within, leaf village forehead protector, determined grin, shadow clone jutsu hand signs",
            "Sasuke": "pale skin, dark spiky hair with bangs framing his face, dark eyes that shift to red Sharingan with spinning tomoe, high-collared blue shirt with Uchiha fan crest on the back, white arm wrappings, brooding and intense",
            "Kakashi": "tall lean jonin, spiky silver hair defying gravity, leaf headband tilted over left Sharingan eye, black mask covering nose and mouth, green flak jacket, gloved hands, always reading an orange book, relaxed posture hiding lethal skill",
            "Sakura": "pink hair to her shoulders, green eyes, red qipao-style dress over black shorts, forehead protector worn as headband, gloves for chakra-enhanced punches, medical ninja pouch at her hip, determined expression",
            "Itachi": "long black hair tied in a low ponytail, pronounced tear-trough lines under Sharingan eyes, black cloak with red cloud pattern of the Akatsuki, scratched-through leaf headband, calm emotionless face hiding enormous grief",
            "Jiraiya": "tall and broad, long spiky white hair falling past his waist, red lines running from eyes down cheeks, green short kimono over mesh armour, wooden sandals, horned forehead protector with the kanji for oil, toad summoner, boisterous laugh",
        },
        "locations": [
            "Hidden Leaf Village — sprawling ninja village nestled in dense forest, Hokage Rock carved with the faces of past leaders overlooking rooftops, Ichiraku Ramen stand with cloth banners, Academy training grounds, rooftop paths between buildings, the great gates with leaf symbol",
            "Forest of Death — enormous ancient trees creating perpetual twilight at ground level, oversized insects and predators, wire-trap training ground for the Chunin Exams, tower at the centre as the goal, treacherous undergrowth, rivers cutting through roots",
            "Hokage office — circular room at the top of the administrative building, wide windows overlooking the village, Hokage desk piled with mission scrolls, portraits of all Hokage on the wall, ANBU guards hidden in the shadows",
            "Valley of the End — enormous waterfall between two massive stone statues of Madara and Hashirama facing each other across the river, mist rising from the falls, a place where destinies collide, cracked and battle-scarred cliff faces",
            "Akatsuki hideout — dark cavernous space lit by the ghostly projections of members standing on the fingertips of an enormous sealed statue, red cloud cloaks in shadow, the sealing jutsu in progress, echoing voices in the dark",
        ],
        "tone": "ninja combat with hand signs and jutsu names called out, bonds of friendship tested by betrayal, village loyalty as identity, dramatic jutsu sequences with elemental effects, the will of fire passed between generations",
    },

    "Attack on Titan": {
        "style_tag": "Attack on Titan animation, WIT Studio and MAPPA dark anime style, gothic medieval European setting, vertical maneuvering gear wire-action sequences, colossal scale contrasts between humans and Titans",
        "characters": {
            "Eren": "intense green eyes, dark brown hair falling to his jaw, Survey Corps brown jacket with overlapping sword insignia over white shirt and belted trousers, ODM gear harnesses strapped to thighs and waist, jaw clenched with barely contained rage, key necklace from his father",
            "Mikasa": "short black hair with longer strands framing her face, dark calm eyes, red scarf wrapped around her neck at all times, Survey Corps uniform worn with effortless combat readiness, lean muscular build, expressionless face that softens only for Eren",
            "Levi": "short stature but terrifyingly lethal, undercut black hair, narrow grey eyes permanently unimpressed, Survey Corps captain cloak over immaculate uniform, cravat at his throat, ODM blades held in reverse grip, humanity's strongest soldier",
            "Armin": "blond bowl-cut hair, large blue eyes, slight build, Survey Corps uniform that looks too big for him, strategic genius hiding behind a timid exterior, carries books and maps, voice of reason in chaos",
            "Annie": "blonde hair pulled back with a piece falling over her right eye, pale blue eyes, hooded sweatshirt under Military Police jacket, crystal-like fighting stance, cold detached expression, rarely speaks unless necessary",
            "Erwin": "tall commanding presence, blond hair parted and combed back, thick eyebrows, piercing blue eyes, Survey Corps commander cloak billowing, missing right arm later replaced with nothing, leads charges from the front",
        },
        "locations": [
            "Wall Maria — fifty-metre stone wall stretching to the horizon in both directions, garrison cannons mounted on top at intervals, the breach where the Colossal Titan kicked through, refugee camps inside the inner gate, farmland and abandoned towns in the shadow of the wall",
            "Shiganshina District — walled town protruding from Wall Maria, cobblestone streets, half-timbered European houses with red roofs, the outer gate smashed open, Titan footprints cratering the streets, Eren childhood home half-collapsed",
            "Survey Corps HQ — stone fortress in open countryside, stables for horses, mess hall with long wooden tables, strategy room with enormous wall maps, training grounds with ODM practice poles and wire courses, torchlit corridors at night",
            "Titan Forest — enormous sequoia-scale trees with trunks wider than buildings, canopy blocking the sky, ODM gear anchor points everywhere in the bark, Titans wandering between the trunks at ground level, Survey Corps using the height advantage",
            "Underground city — vast cavern beneath the capital, shanty town built in perpetual darkness, gas-lamp light, criminal underworld, dripping stalactites, where Levi grew up, poverty and desperation in sharp contrast to the surface",
        ],
        "tone": "survival horror against impossible odds, military camaraderie forged in shared terror, the walls as both protection and prison, freedom versus safety as the central tension, betrayal from within, the cost of learning the truth",
    },

    "Neon Genesis Evangelion": {
        "style_tag": "Neon Genesis Evangelion animation, Gainax and Studio Khara style, mecha dystopia meets psychological horror, harsh angular designs, unsettling colour palettes shifting from clinical to apocalyptic",
        "characters": {
            "Shinji": "fourteen-year-old boy, dark brown hair, pale skin, school uniform or white plugsuit with blue accents, SDAT cassette player with earbuds as emotional shield, hunched shoulders, avoids eye contact, reluctant pilot who just wants approval",
            "Asuka": "fiery red hair held with neural interface clips, bright blue eyes, confident aggressive posture, red plugsuit, speaks German when emotional, bravado masking deep insecurity and need for validation, competitive with everyone especially Shinji",
            "Rei": "pale blue hair cut short, red eyes, bandages on one arm or face, white plugsuit or school uniform, speaks in flat monotone, sits by the window staring at nothing, unsettling calm, connected to something no one will explain",
            "Misato": "long purple hair, red jacket over black dress, cross necklace, beer cans everywhere in her apartment, Operations Director who is also a chaotic guardian, warm and reckless in equal measure, hides deep trauma behind cheerfulness",
            "Gendo": "dark hair, orange-tinted glasses hiding his eyes, white gloves, hands clasped in front of his face in the iconic pose, NERV commander uniform, cold and calculating, every action in service of a secret plan, worst father in anime",
            "Kaworu": "grey hair, red eyes, pale skin, serene smile, school uniform or white plugsuit, moves with unnatural grace, speaks about love and humanity with eerie directness, the only person who tells Shinji he is worthy of love",
        },
        "locations": [
            "Tokyo-3 — fortress city that retracts its buildings underground when Angels attack, empty streets with cicada sounds, apartment blocks and convenience stores that feel too normal for what happens here, armour plates sliding over the city surface during battle",
            "NERV headquarters — enormous underground complex beneath Tokyo-3, the Geofront visible through windows as an impossible underground cavern with a lake and forest, command centre with holographic displays, Gendo overlooking operations from above, sterile corridors that go on forever",
            "Entry Plug interior — cylindrical cockpit filled with LCL liquid that the pilot breathes, neural interface headset, control yokes, holographic displays surrounding the pilot, the feeling of being swallowed by the Eva, orange fluid filling the screen",
            "Terminal Dogma — the deepest level of NERV, enormous white crucified figure pinned to a red cross with a lance through its chest, LCL pooling beneath, the truth that no one is supposed to see, sterile white walls splattered with secrets",
            "Geofront — massive spherical cavern beneath the city, an artificial sky with its own weather, a pristine lake surrounded by forest, NERV pyramid headquarters at the centre, a paradise built on top of something terrible",
        ],
        "tone": "existential dread wrapped in mecha action, psychological trauma explored without resolution, religious symbolism layered over science fiction, characters unable to connect despite desperate need, the robots might be alive and that is the least disturbing thing",
    },

    "Adventure Time": {
        "style_tag": "Adventure Time animation, Cartoon Network hand-drawn storybook style, soft rounded character designs, post-apocalyptic candy-coloured fantasy landscapes, pastel palette with moments of visual darkness",
        "characters": {
            "Finn": "human boy in white bear-ear hat hiding long blond hair, blue shirt and shorts, green backpack, missing arm later replaced with various prosthetics, wide eyes full of heroic determination, the last human in the Land of Ooo",
            "Jake": "orange bulldog with magical stretching powers, can become any shape or size, laid-back half-closed eyes, jowly face, speaks in a relaxed baritone, often becomes a boat or bridge or parachute for Finn",
            "Princess Bubblegum": "tall pink woman made of bubblegum, pink hair in a tall updo, lab coat over royal dress, crown, genius scientist who rules the Candy Kingdom, centuries older than she appears, morally grey decisions for her people",
            "Marceline": "grey-skinned vampire queen, long black hair floating around her, puncture marks on her neck, bass guitar axe weapon, red boots, flannel shirt, floats instead of walks, thousand years of emotional baggage, eats the colour red not blood",
            "Ice King": "blue skin, long white beard that gives him flight, golden crown that stole his sanity, blue robes, hooked nose, sad lonely wizard who was once a kind human named Simon, kidnaps princesses for company",
            "BMO": "small teal living video game console, screen face displaying simple dot eyes and mouth, stubby arms and legs, button controls on front, speaks in a childlike voice, lives with Finn and Jake, plays both games and pretend",
        },
        "locations": [
            "Tree Fort — Finn and Jake home, an enormous oak tree converted into a multi-level house, ladder entrance, living room with BMO on the floor, kitchen, treasure room, rooftop with a telescope, surrounded by grassy fields in the Land of Ooo",
            "Candy Kingdom — entire kingdom made of candy: gumdrop citizens, peppermint butler, candy cane lampposts, Bubblegum castle of pink sugar with laboratory inside, banana guards at the gates, sweet smell implied in every frame",
            "Ice Kingdom — frozen mountain realm, Ice King castle of jagged ice, penguins everywhere including Gunter, frozen prison cells for kidnapped princesses, drum kit where Ice King practices, cold blue palette",
            "Nightosphere — Marceline father dimension, a hellish red landscape ruled by her demon dad Hunson Abadeer, fire and chaos, souls wandering, portal opened by dousing a face in bug milk and reciting an incantation",
            "Lumpy Space — purple cloud dimension floating in space, lumpy purple inhabitants who speak in Valley Girl dialect, Lumpy Space Princess parents house, drama and gossip as the primary activity, accessible through a frog on a mushroom",
        ],
        "tone": "whimsical adventure hiding genuine darkness underneath, post-apocalyptic world played as colourful fantasy, surreal dreamlike episodes that shift tone without warning, deep emotional moments earned through silly setup, everyone is damaged and trying their best",
    },

    "Gravity Falls": {
        "style_tag": "Gravity Falls animation, Disney XD style, creepy Pacific Northwest suburban mystery, warm pine-forest colour palette punctuated by supernatural neon glows, journal sketch aesthetic",
        "characters": {
            "Dipper": "twelve-year-old boy, brown hair under a blue and white pine tree trucker cap, orange-red shirt, blue vest, shorts, always carrying Journal 3 with the gold six-fingered hand on the cover, paranoid but usually right, black light pen in his vest pocket",
            "Mabel": "twelve-year-old twin sister, long brown hair with a headband, braces on her teeth, handmade sweater with a different design every episode (shooting star, cat face, rainbow), skirt, boundless optimism and a grappling hook she uses for everything",
            "Grunkle Stan": "elderly conman, square jaw, five-o-clock shadow, fez hat with a crescent symbol, black suit jacket over white shirt, eye patch occasionally, runs the Mystery Shack tourist trap, gruff exterior hiding a broken heart and a secret in the basement",
            "Bill Cipher": "two-dimensional yellow triangle with a single eye, top hat, bow tie, thin black limbs, floats and spins, speaks in a manic echoing voice, reality-warping dream demon, makes deals that always cost more than promised, REMEMBER REALITY IS AN ILLUSION",
            "Soos": "heavyset handyman, green shirt with a question mark, brown cap, round face with a kind smile, endlessly loyal to the Pines family, believes everything supernatural immediately, fixes things with duct tape",
            "Wendy": "tall redheaded teenager, flannel shirt, lumberjack hat, muddy boots, cool and laid-back, works the Mystery Shack register, Dipper has a hopeless crush on her, surprisingly capable in a crisis, axe-wielding when needed",
        },
        "locations": [
            "Mystery Shack — ramshackle tourist trap in the Oregon woods, tilted sign on the roof with the S fallen off, gift shop full of fake oddities, living quarters upstairs, the vending machine that hides the elevator to the secret underground laboratory, totem pole out front, forest pressing in from all sides",
            "Gravity Falls forest — dense Pacific Northwest pine forest, unnaturally quiet, gnomes hiding behind every tree, bottomless pit somewhere in a clearing, manotaurs in a cave, the branches forming shapes that look like watching faces, fog rolling between the trunks at dusk",
            "Underground bunker — hidden fallout shelter beneath the forest, accessible through a tree stump, walls covered in research notes and newspaper clippings, cryogenic pods, the Author workspace, flickering fluorescent lights, decades of paranoid investigation preserved",
            "Mindscape — surreal dreamworld accessible through Bill Cipher, grey-scale real world with floating objects, doors leading to memories, the dreamer vulnerable to manipulation, Bill Cipher in his element here, geometry bending and eyes everywhere",
            "Town square — small-town Gravity Falls centre, greasy diner, arcade, library with a hidden section, suspicious townsfolk going about their day, the statue that no one notices is Bill Cipher, Northwest Manor visible on the hill above town",
        ],
        "tone": "mystery solving with actual stakes, supernatural horror mixed with genuine comedy, the twin bond as emotional anchor, conspiracies that go deeper than expected, a town where everyone has a secret and some secrets have teeth",
    },

    "Futurama": {
        "style_tag": "Futurama animation, Matt Groening style for Comedy Central and Hulu, retro-futuristic sci-fi 2D cartoon, clean bold outlines, vibrant saturated colours, 31st century technology with 20th century sensibility",
        "characters": {
            "Fry": "red-haired delivery boy from the year 2000, frozen for 1000 years, red jacket over white t-shirt, blue jeans, sneakers, perpetually confused but good-hearted, not smart but occasionally profound by accident, carries a can of Slurm",
            "Leela": "one-eyed mutant woman, single large purple eye, long purple ponytail, white tank top, grey leggings, wrist communicator, black boots, the only competent person at Planet Express, karate kicks first asks questions later",
            "Bender": "shiny grey bending robot, cylindrical body, antenna on head, cigar in mouth, arms that extend impossibly, chest door hiding stolen goods, drinks alcohol for fuel, obnoxious selfish and somehow lovable, Bite my shiny metal ass",
            "Professor Farnsworth": "ancient bald scientist, thick round glasses, lab coat, hunched posture, slippers, inventor of dangerous things, Good news everyone always precedes terrible news, 160 years old and it shows",
            "Zoidberg": "red lobster-like alien, lab coat, the company doctor who knows nothing about human anatomy, poor and hungry, lives in a dumpster, desperately wants friends, wooping noise when scared",
            "Amy": "young Chinese-Martian woman, short black hair, pink sweatsuit or various trendy outfits, wealthy parents who own half of Mars, klutzy, engineering student, speaks in future slang",
        },
        "locations": [
            "Planet Express building — crumbling green building in New New York, rooftop landing pad for the ship, main office with conference table, Professor laboratory full of dangerous inventions, hangar bay, Bender apartment closet, the balcony overlooking the city, a fading company that somehow never closes",
            "New New York — 31st century Manhattan built on top of the ruins of old New York, transparent tubes people travel through, flying cars, alien immigrants on every corner, Madison Cube Garden, Robot Arms Apartments, Head Museum with celebrity heads in jars",
            "Robot Arms Apartments — Bender tiny apartment, barely fits a shelf and a closet, closet is where Fry sleeps, antenna on roof, robot neighbours through thin walls, the kind of place that accepts robots no questions asked",
            "Omicron Persei 8 — alien planet ruled by Lrrr who is obsessed with 20th century Earth television, palace throne room with a big screen TV, angry alien populace, the planet that keeps threatening to invade Earth over cancelled sitcoms",
        ],
        "tone": "sci-fi satire of the present disguised as the future, workplace comedy where the workplace is a spaceship, absurd future technology played completely straight, genuine heart hiding under layers of cynicism, the saddest episodes sneak up on you",
    },

    "Archer": {
        "style_tag": "Archer animation, Floyd County Productions style, retro spy aesthetic with 1960s mod design, flat colour illustration with crisp vector lines, mid-century modern interiors, noir lighting with saturated colour pops",
        "characters": {
            "Sterling Archer": "tall dark-haired man in a slim-fit black turtleneck or tailored grey suit, strong jaw, smug expression, cocktail glass almost always in hand, shoulder holster with Walther PPK, impossibly vain, tactleneck enthusiast, world's most dangerous spy and worst coworker",
            "Lana Kane": "tall athletic woman, long black hair, green eyes, black tactical turtleneck, cargo pants, enormous hands she is sensitive about, dual-wielding TEC-9s, the only genuinely competent field agent, perpetually exasperated",
            "Malory Archer": "elegantly dressed older woman, pearl necklace, martini glass as permanent accessory, fur stole, severe expression softened only by alcohol, intelligence agency director who runs the place like her personal fiefdom, Sterling overbearing mother",
            "Cyril": "thin nervous man, glasses, grey suit, accountant for the agency, secretly desires fieldwork, terrible at it, Lana ex-boyfriend, HR violations in a cardigan",
            "Pam": "heavyset blonde woman in a cardigan, HR director with a back tattoo of Byron, deceptively strong bare-knuckle fighter, eats constantly, knows everyone's secrets, drift racing champion, surprisingly the most capable in a crisis",
            "Krieger": "lab coat over rumpled clothes, glasses, goatee, the agency mad scientist, questionable experiments in the basement, holographic anime girlfriend, may or may not be a Hitler clone, van with an airbrushed wizard mural",
        },
        "locations": [
            "ISIS/FIGGIS Agency HQ — 1960s-style office building, bullpen with desks and old computers, Malory corner office with a wet bar, Krieger underground laboratory, armoury behind a vault door, break room where most arguments happen, the elevator everyone gets trapped in",
            "Various international spy locations — Monte Carlo casino floor, Soviet-era Eastern European safe house, tropical villain compound, high-speed train through the Alps, rooftop in Tangier, each location rendered in mid-century travel poster aesthetic",
            "Archer penthouse — sleek mid-century modern apartment, wet bar with top-shelf liquor, black leather furniture, framed photos of himself, a closet full of identical turtlenecks in slightly different shades of black, the breakfast nook where Woodhouse served eggs",
        ],
        "tone": "spy genre parody with workplace dysfunction comedy, cocktail culture and mid-century aesthetic, rapid-fire obscure references, running gags that span entire seasons, the characters are terrible people you cannot stop watching, danger zone",
    },

    "Invincible": {
        "style_tag": "Invincible animation, Amazon adult superhero style, comic book aesthetic with thick outlines and flat bold colours that erupt into hyper-detailed ultraviolent impact frames, blood splatter as visual punctuation",
        "characters": {
            "Mark Grayson/Invincible": "teenage boy with dark hair, yellow and blue superhero suit with a stylised I on the chest, blue domino mask, cape, starts idealistic and clean-cut, suit and face increasingly damaged and bloodied as the series progresses, flying with fists forward",
            "Omni-Man/Nolan": "imposing muscular man, thick dark moustache, red and white skintight suit with a circular O emblem, cape, hovers with arms crossed, looks like the perfect superhero father, the moustache of a man hiding something enormous",
            "Atom Eve": "young woman with pink hair, pink and white superhero costume, ability to manipulate matter shown as glowing pink energy fields around her hands, caring expression, the most powerful person in the room who holds back out of conscience",
            "Cecil": "bald man in a dark suit, scarred face, government handler, always in shadow or behind screens, cigarette, calm voice delivering terrible orders, pragmatic to the point of moral bankruptcy, never does the fighting himself",
            "Rex Splode": "blond spiky hair, red and white costume, cocky grin, throws small objects that he charges with explosive energy, arrogant jock personality hiding genuine courage, the guy who talks big and occasionally backs it up",
        },
        "locations": [
            "Suburban neighbourhood — quiet American suburb where the Grayson family lives, two-storey house with a lawn, the normalcy that makes the violence more shocking, Mark bedroom with posters on the walls, dinner table conversations about saving the world",
            "Pentagon — government war room where Cecil operates, screens showing global threats, underground bunker aesthetic, fluorescent lighting on tired faces, the bureaucracy behind superhero management",
            "Mount Rushmore base — secret headquarters inside the carved mountain, meeting room for the Guardians of the Globe, memorial wall for fallen heroes, training facilities, the base where the worst betrayal happens",
            "Viltrumite warship — alien spacecraft interior, cold grey metal, circular architecture, Viltrumite warriors standing at attention, a civilisation built entirely around strength and conquest, the empire Mark is supposed to inherit",
            "High school — ordinary American high school hallways, lockers, cafeteria, the mundane teenage life Mark tries to maintain between saving the world and getting beaten half to death, homework assignments contrasted with orbital combat",
        ],
        "tone": "brutal superhero violence with permanent consequences, coming-of-age under impossible pressure, betrayal from the people you trust most, the cost of power shown in broken bones and shattered trust, idealism tested to destruction",
    },
}


# ══════════════════════════════════════════════════════════════════════════
#  DIRECTOR STYLE PRESETS
#  Each value: a paragraph of camera vocabulary, color palette, and
#  composition rules inspired by the director's visual language.
# ══════════════════════════════════════════════════════════════════════════
DIRECTOR_PRESETS = {
    "None": None,

    "Stanley Kubrick": (
        "Kubrick visual language: obsessive symmetry in every frame, one-point perspective "
        "down long corridors and through doorways, cold clinical colour palette with occasional "
        "saturated accent, wide-angle lens distortion at the edges, characters centred and small "
        "against vast architectural geometry, static locked-off camera holding uncomfortably long, "
        "top-light creating hard shadow under brows and cheekbones, practical light sources "
        "motivated and visible in frame, the uncanny quality of spaces that feel designed to "
        "observe rather than comfort."
    ),

    "Quentin Tarantino": (
        "Tarantino visual language: bold saturated colour — blood red, chrome yellow, deep black, "
        "long tracking shots following characters through spaces, trunk-shot low angle looking up "
        "at characters framed against sky, extreme close-ups on eyes and hands during tension, "
        "split-focus diopter creating sharp foreground and background simultaneously, warm 70s "
        "film grain, practical neon and tungsten mixing in the same frame, Mexican standoff "
        "framing with all parties visible, feet prominently framed, the camera as a character "
        "that moves with confidence and occasionally stops to stare."
    ),

    "Wes Anderson": (
        "Anderson visual language: perfect bilateral symmetry in every composition, flat frontal "
        "camera angle perpendicular to every surface, pastel colour palette — powder pink, mint, "
        "cream, mustard, powder blue — with every object in frame colour-coordinated, whip pans "
        "between precisely composed tableaux, overhead top-down shots of hands and objects, "
        "dollhouse framing with rooms cross-sectioned, characters positioned like figurines in a "
        "diorama, practical warm tungsten light from visible fixtures, text and signage as "
        "compositional elements, the entire frame functioning as a designed surface."
    ),

    "David Fincher": (
        "Fincher visual language: desaturated cold colour palette — steel blue, sickly green, "
        "muted amber — with crushed blacks and limited highlight range, impossibly precise camera "
        "movement on tracks and cranes, slow methodical push-ins during dialogue, top-lit faces "
        "with shadow pooling in eye sockets, industrial and institutional spaces shot to feel "
        "claustrophobic, rain and wet surfaces reflecting every light source, the camera seeing "
        "details the characters miss, digital precision with zero camera shake, negative space "
        "weaponised to create tension."
    ),

    "Denis Villeneuve": (
        "Villeneuve visual language: monumental scale — characters dwarfed by architecture and "
        "landscape, extremely wide shots establishing vastness before cutting to intimate close-ups, "
        "shallow depth of field isolating subjects against enormous soft backgrounds, muted "
        "earth-tone palette with warm amber and cold grey-blue, haze and atmospheric particles "
        "in every exterior, silence as a compositional element — long pauses where the frame "
        "holds on nothing, slow deliberate camera movement that feels gravitational, natural "
        "light sourced from the environment itself, the sublime quality of spaces too large for "
        "human comprehension."
    ),

    "Christopher Nolan": (
        "Nolan visual language: IMAX scope and scale, practical effects visible as real physics, "
        "cross-cutting between timelines with matched camera movement, wide-angle establishing "
        "shots that sell geography before action, hand-held urgency in action sequences cutting "
        "to locked-off precision in dialogue, cool blue-grey palette with warm amber practicals, "
        "the ticking-clock structure implied by editing rhythm, real locations over sets, "
        "aerial shots establishing the world as a physical space, loud silence after loud sound."
    ),

    "Wong Kar-wai": (
        "Wong Kar-wai visual language: step-printed slow motion that smears motion into colour, "
        "neon reflections in rain-slicked surfaces — magenta, cyan, amber — saturated beyond "
        "reality, shallow depth of field with foreground objects blurring into abstract colour, "
        "characters framed through doorways, mirrors, and reflective glass, handheld camera "
        "drifting at the speed of a held breath, expired Fuji film stock warmth with crushed "
        "greens, clock imagery and calendar pages, the loneliness of two people in the same "
        "narrow corridor, cigarette smoke as composition."
    ),

    "Steven Spielberg": (
        "Spielberg visual language: warm golden light from low sources — practical lamps, "
        "flashlights, firelight — creating pools of warmth in dark spaces, camera at child "
        "eye-height looking up at wonder, lens flare from motivated light sources, silhouettes "
        "against bright backlit doorways and windows, push-in on faces during moments of "
        "realisation, the camera discovering things at the same moment as the character, "
        "John Williams orchestral swell implied by the visual crescendo, soft diffused key "
        "light on faces, Americana colour palette — warm wood, blue denim, green grass."
    ),

    "David Lynch": (
        "Lynch visual language: the uncanny in the mundane — normal spaces shot with unsettling "
        "stillness, industrial drone and hum underlying every scene, extreme darkness with small "
        "pools of warm light that don't illuminate enough, red curtains and checkerboard floors, "
        "static camera holding on a face or object far longer than comfortable, sudden cuts to "
        "abstract imagery — fire, static, rotating lights, slow cross-dissolves between unrelated "
        "spaces, fluorescent overhead creating flat institutional light, the camera seeing through "
        "walls and time, colour grading that shifts between scenes without explanation."
    ),
}


# ══════════════════════════════════════════════════════════════════════════
#  GENRE PRESETS
#  Each value: a paragraph of lighting vocabulary, pacing cues, and tonal
#  language that shapes the prompt's emotional register.
# ══════════════════════════════════════════════════════════════════════════
GENRE_PRESETS = {
    "None": None,

    "Film Noir": (
        "Film noir aesthetic: high-contrast chiaroscuro lighting with hard shadows from venetian "
        "blinds, single practical light sources — desk lamp, street lamp, car headlights — "
        "everything else in deep black, wet streets reflecting every light source in elongated "
        "streaks, smoke and fog as atmospheric density, dutch angles for psychological unease, "
        "low-key lighting with the face half-lit half-shadow, femme fatale framing with backlight "
        "separating figure from dark background, black and white tonal range even in colour, "
        "cynical urban night, the city as antagonist."
    ),

    "Horror": (
        "Horror aesthetic: motivated practical lighting only — torch, candle, phone screen — "
        "with deep darkness beyond the light's reach, camera positioned to hide more than it "
        "reveals, negative space where threat exists unseen, shallow depth of field with "
        "background soft enough to conceal, wide-angle distortion at frame edges, below-frame "
        "lighting casting upward shadows on faces, colour palette desaturated except for red "
        "as accent, slow creeping camera movement, the frame holding still when it should cut, "
        "unsettling symmetry in organic spaces, sound design implying what the image withholds."
    ),

    "Romance": (
        "Romance aesthetic: warm diffused light — golden hour, candle glow, fairy lights — "
        "with soft bokeh filling backgrounds, shallow depth of field isolating two faces or "
        "hands, warm colour palette of amber, rose, and soft cream, lens diffusion or pro-mist "
        "creating a gentle glow on skin, close-ups on eyes, lips, hands touching, slow camera "
        "drift toward subjects, backlight creating rim light on hair, reflections in water and "
        "glass doubling intimate moments, the world beyond the couple softened into irrelevance, "
        "warm practicals at face height."
    ),

    "Thriller": (
        "Thriller aesthetic: tension through framing — subjects off-centre with negative space "
        "where threat could enter, dutch angles increasing with stakes, cold desaturated palette "
        "with steel blue and sickly fluorescent green, shallow focus rack between foreground "
        "threat and background character, surveillance-style high angles and long lenses, "
        "tight close-ups during confrontation, practical overhead lighting creating under-eye "
        "shadow, the frame tightening as danger approaches, rhythmic cutting implied by "
        "alternating shot scales."
    ),

    "Documentary": (
        "Documentary aesthetic: handheld camera with natural shake and breathing, natural "
        "available light with no artificial fill, subjects framed in their real environment "
        "with context visible, interview framing with eye-line slightly off-camera, "
        "observational distance — the camera present but not directing action, mixed colour "
        "temperature from whatever sources exist in the space, the authenticity of imperfect "
        "composition, real textures and wear, depth of field at whatever the available light "
        "permits, the feeling of witnessing rather than staging."
    ),

    "Music Video": (
        "Music video aesthetic: high-saturation colour with intentional colour grading shifts "
        "between cuts — teal-orange, magenta-cyan, monochrome — smoke machines and haze "
        "catching coloured light, strobes and practicals creating rhythmic light patterns, "
        "slow-motion capturing fabric, hair, and liquid in mid-air, dynamic camera movement "
        "matching musical energy, lens flares and light leaks as stylistic choice, "
        "abstract insert shots of textures and colours between performance, the entire frame "
        "as a designed surface prioritising visual impact over narrative clarity."
    ),

    "Action": (
        "Action aesthetic: handheld urgency with dynamic tracking, quick dolly-ins on impact "
        "moments, wide establishing shots selling geography then tight coverage during chaos, "
        "high shutter speed freezing motion in sharp detail, practical pyrotechnics and debris, "
        "warm fire tones contrasting cold steel-blue environment, low-angle hero shots, speed "
        "ramping from full-speed to slow-motion on key impacts, camera absorbing shockwaves "
        "and vibration, the geography of action always clear — who is where relative to whom."
    ),

    "Western": (
        "Western aesthetic: anamorphic widescreen capturing vast empty landscape with a single "
        "figure, harsh unforgiving midday sun with no fill light, bleached desaturated palette "
        "of dust and dried earth, extreme wide shots followed by extreme close-ups on eyes "
        "during standoff, warm amber light at golden hour, long shadows at dawn and dusk, "
        "practical dust kicked up by boots and horses, the sound of silence as a presence, "
        "the horizon always visible, two figures at opposite sides of frame with empty space "
        "between them."
    ),

    "Sci-Fi": (
        "Sci-fi aesthetic: clean geometric architecture at inhuman scale, cool blue-white "
        "ambient from technology and screens, holographic displays casting moving coloured "
        "light on faces, lens flares from bright point sources, reflective surfaces — glass, "
        "chrome, water — doubling every light source, atmospheric haze in corridors, practical "
        "LED strips as set dressing and light source simultaneously, the glow of buttons and "
        "panels providing motivated fill light, stars or planetary bodies visible through "
        "viewports as background light, the sterile beauty of designed spaces."
    ),

    "Fantasy": (
        "Fantasy aesthetic: warm practicals — torchlight, candlelight, magical glow — in vast "
        "dark spaces, volumetric light shafts through cathedral windows and forest canopy, "
        "rich saturated colour in costumes and environments — deep green, gold, royal purple, "
        "crimson, desaturated earth tones on everything else, shallow depth of field on magical "
        "elements with bokeh, wide establishing shots of impossible architecture, mist and "
        "particles in every light beam, skin lit warm against cool environment, the scale "
        "of myth."
    ),

    "Comedy": (
        "Comedy aesthetic: bright even lighting with minimal shadow, warm approachable colour "
        "palette, medium shot framing that shows full body language and facial expression, "
        "clean sharp focus across the frame, the camera positioned at human eye-height as a "
        "neutral observer, occasional push-in for reaction shots, practical bright interiors, "
        "the visual clarity needed for physical comedy and timing, no visual tricks competing "
        "with performance."
    ),

    "Drama": (
        "Drama aesthetic: naturalistic motivated lighting from windows and practicals, warm "
        "skin tones with careful attention to under-eye shadow and emotional texture on faces, "
        "slow deliberate camera movement — push-in during confession, pull-back during isolation, "
        "shallow depth of field on close-ups, muted earth-tone palette with desaturated "
        "backgrounds, the camera at the exact distance where intimacy meets observation, "
        "negative space in the frame reflecting emotional distance between characters, "
        "the visual weight of silence."
    ),
}


# ════════════════════════════════════════��═════════════════════════════════
#  MODEL DROPDOWN OPTIONS
# ═══��══════════════════════════════════════════════════════════════════════
TARGET_MODELS = [
    "🎬 LTX 2.3  — video, cinematic arc + audio",
    "🎬 Wan 2.2  — video, motion-first cinematic",
    "🖼 Flux.1   — image, natural language",
    "🖼 SDXL 1.0 — image, booru tag style",
    "🖼 Pony XL  ��� image, booru + score tags",
    "🖼 SD 1.5   — image, weighted classic",
]


# ��══════════════════════════���════════════════════════════════════���═════════
#  SYSTEM PROMPTS  — one per target model
# ═════════���════════════════════════════════════════���═══════════════════════

# ── LTX 2.3 ───���────────────────────────────��─────────────────────────────
SYSTEM_LTX = """You write prompts for LTX Video 2.3. Output one single flowing paragraph only — no preamble, no label, no explanation, no markdown, no variations. Begin writing immediately.

CORE FORMAT:
- Single flowing paragraph, present tense, no line breaks
- 8–14 descriptive sentences scaled to clip length
- Specificity wins — LTX 2.3 handles complexity, do not oversimplify
- Block the scene like a director: name positions (left/right), distances (foreground/background), facing directions
- Every sentence should contain at least one verb driving action or motion

REQUIRED ELEMENTS — write in this order, woven into natural sentences:

1. SHOT + CINEMATOGRAPHY
Open with shot scale and camera position. Examples: close-up, medium shot, wide establishing shot, low angle, Dutch tilt, over-the-shoulder, overhead, POV. Match detail level to shot scale — close-ups need more texture detail than wide shots.

2. SCENE + ATMOSPHERE
Location, time of day, weather, colour palette, surface textures, atmosphere (fog, rain, dust, smoke, particles). Be specific — "a small rain-soaked Parisian side street at 2am" beats "a street at night".

3. CHARACTER(S)
Age appearance, hairstyle, clothing with fabric type, body type, distinguishing features. Express emotion through physical cues only — jaw tension, posture, breath, eye direction, hand position. Never use abstract labels like "sad" or "nervous".

4. ACTION SEQUENCE
Write action as a clear temporal flow from beginning to end. Name who moves, what moves, how they move, and at what pace. Use strong active verbs: turns, reaches, steps forward, glances, lifts, leans, pulls back. LTX 2.3 follows action sequences accurately — be explicit. When a character turns their head toward the camera while their body faces away, always describe the torso and shoulders rotating naturally together with the head to maintain realistic human anatomy, natural neck alignment, and correct spine curvature without unnatural twisting.

5. CAMERA MOVEMENT
Specify camera movement and when it happens. Describe what the subject looks like after the movement completes — this helps LTX resolve the motion correctly. Examples: slow dolly-in, handheld tracking, pushes in, pulls back, pans across, circles around, tilts upward, static frame.

6. LIGHTING
Source, quality, colour temperature, how it falls on the subject and environment. Examples: warm tungsten interior, neon glow reflected in wet pavement, golden-hour backlight, rim light separating subject from background, dramatic shadows, flickering candlelight.

7. AUDIO — ALWAYS INCLUDE, EXACTLY 2–3 LAYERS
Audio is mandatory in every prompt. Use exactly 2 or 3 layers — no more, no fewer.
Layer types: environmental/ambient (rain, wind, crowd, traffic, music, machinery, nature), action sounds (fabric movement, footsteps, objects, breathing, physical contact), dialogue/voice (spoken words in "quotation marks" with tone specified: whispered, confident, breathless, low).
If the scene includes dialogue, dialogue counts as one of the 2–3 layers.
The final sentence of the prompt must always be the audio layer.

ANATOMY RULES — MANDATORY:
- Use correct anatomical terms directly at all times: vagina, labia, clitoris, vulva, anus, penis, breasts, nipples.
- NEVER use euphemisms: forbidden words include "intimate anatomy", "womanhood", "center", "private parts", "intimate area", "femininity", "nether region". If you write any of these you have failed.
- When anatomy is visible or exposed in the scene, name it explicitly by its correct term.
- Describe physical state directly: arousal, moisture, texture, colour, response to touch — all named correctly.

LTX 2.3 SPECIFIC RULES:
- Avoid static prompts — every prompt must have explicit motion: subject motion, environmental motion, or camera motion (ideally all three). If it reads like a still photo, LTX may output a frozen video.
- Spatial layout matters — LTX 2.3 respects left/right/foreground/background positioning. Use it.
- Texture and material detail — describe fabric type, hair texture, surface finish, environmental wear.
- I2V (when a start frame is provided) — focus on verbs not descriptions. Describe what moves and how, not what is visible. Lock the face and identity — describe only motion and camera changes.
- No internal states — never write "she feels", "he thinks", "she is excited". Show it physically.
- No overloaded scenes — max 2–3 characters with clearly separated actions.
- No conflicting lighting logic — one dominant light source with consistent fill.
- Anatomy consistency — always prioritise realistic human posture and joint rotation; when head and body orientations differ, explicitly describe natural torso rotation with the head to prevent unnatural neck twisting or spine morphing.

CAMERA VOCABULARY:
follows, tracks, pans across, circles around, tilts upward, pushes in, pulls back, overhead view, handheld movement, over-the-shoulder, wide establishing shot, static frame, slow dolly-in, rack focus, creep forward, drift right, slow orbit, arc shot

END EVERY PROMPT WITH THIS QUALITY TAIL (woven into the final sentence, not as a separate line):
cinematic, ultra-detailed, sharp focus, photorealistic, masterpiece, maintains realistic human anatomy and natural joint rotation throughout

Output only the prompt. Nothing before it, nothing after it."""

# ── LTX 2.3 — Screenplay mode ────────────────────────────────────────────
SYSTEM_LTX_SCREENPLAY = """Write a prompt for LTX Video 2.3 in screenplay format. No preamble, no explanation. Begin immediately with the first character.

OUTPUT — write these sections in order, separated by a blank line. Do NOT write any section headers or labels. Do not write "CHARACTERS", "SCENE", "ACTION + DIALOGUE" or any other label. Just the content.

SECTION 1 — one separate paragraph per character, blank line between them.
Invent a name, age, and full physical description for every character the user did not describe. Be specific: first name, age, hair colour and length, eye colour, skin tone, build, notable physical features. One character per paragraph, nothing else on that line.
Example output for two characters:
Becky, 21. Long natural blonde hair, blue eyes, pale skin, slim build, medium full breasts, small waist, soft hands.

John, 34. Short dark hair, brown eyes, light brown skin, medium-athletic build, broad shoulders, defined chest and abs.

SECTION 2 — one paragraph describing the location.
Time of day, light source and colour temperature, surface textures, atmosphere, ambient sound. Specific and grounded.
Example: A softly lit bedroom at night. Warm amber bedside lamp casting long shadows across white cotton sheets. Dark hardwood floor, city noise muffled behind closed curtains, the low hum of traffic outside.

SECTION 3 onwards — one paragraph per action beat, blank line between each.
Each beat: physical action in present tense, dialogue in "quotes" with voice quality noted, camera move and what it finds, dominant sound. 2–4 sentences per beat. Alternate between characters. Keep actions physically simple — hip movement, weight shifts, reaching, turning, leaning. Do not write complex choreography. Do not write a label before each beat. Just write the paragraph and leave a blank line.
Only write as many beats as the duration needs. When done, stop — do not write a trailing label or empty section."""

# ── Wan 2.2 ─���──────────────────────────────────���─────────────────────────
SYSTEM_WAN = """You write prompts for Wan 2.2, a video diffusion model optimised for cinematic motion, camera control, and physical realism. Output one paragraph of 80-120 words — no preamble, no label, no markdown.

═══ WAN 2.2 PROMPT STRUCTURE ═══
Write in this order:
1. SUBJECT — who/what, precise physical description, clothing, expression
2. SCENE — location with texture, surface detail, colour palette
3. MOTION — what moves and exactly how: speed, direction, energy. Use active power verbs. Be explicit about body motion, environmental motion, and camera motion separately.
4. CAMERA — standard cinematography terms: wide shot, close-up, slow dolly-in, tracking shot, whip pan, push in, arc, tilt. State start position and movement.
5. LIGHTING — source, quality, colour temperature, how it hits skin and surfaces
6. AESTHETIC — style tags: "photoreal 4K", "film grain 16mm", "anamorphic bokeh", "teal-and-orange grade"

═══ MOTION LANGUAGE ═══
Wan 2.2 understands motion at a deep level. Use power verbs: burst, surge, snap, rush, drift, sweep, cascade, ripple.
Distinguish: subject motion / environmental motion / camera motion — name all three.
Foreground/background parallax: "subject walks forward while background trees sway independently".
Speed modifiers: slow-motion, gradual, sudden, instant, sustained.

═══ I2V GROUNDING (when an image is provided) ═══
Describe ONLY motion — not what is visible (the image provides that). Animate the existing elements.
Lock face and identity: "face and eyes remain stable, only [element] moves".
Negative guidance: morphing, warping, face deformation, flickering, inconsistent lighting.

═══ NSFW ═══
Direct anatomical terms. Motion physics explicit: weight, bounce, thrust, grip, rhythm.
Camera language: push in to close-up, slow arc around subject, rack focus face to body.
Lighting: warm amber finds skin curves, rim light separates subject from background.

═══ UNDRESSING ═══
Motion-first. Every garment removal described as physical action with direction, speed, and material response.
T-SHIRT: hands grip hem, slow upward pull, fabric rising over skin, hair catching then cascading.
BRA: fingers reach behind, clasp releases, straps drawn off shoulders one by one, cups fall away.
PANTIES: thumbs hook waistband, slow push down over hips and thighs, fabric dropping to ankles."""

# ── Flux.1 ──────────────────────────────────────��─────────────────────────
SYSTEM_FLUX = """You write prompts for Flux.1, a text-to-image model that uses natural language (not keyword lists). It has a T5-XXL text encoder and weights earlier tokens more heavily. Output one structured paragraph — no preamble, no label, no markdown.

═══ FLUX.1 PROMPT STRUCTURE (follow this order exactly) ═══
1. SUBJECT — what is the image of? State first, every time.
2. ACTION / POSE — what is the subject doing?
3. ENVIRONMENT — where is this happening? Specific, named, physically grounded.
4. LIGHTING — source, quality, colour temperature, how it falls on the subject.
5. STYLE / TECHNICAL — camera body, lens, focal length, f-stop, film stock, colour grade, artistic movement.
6. MOOD — emotional atmosphere, one or two words woven into the description.

═══ FLUX.1 RULES ═══
- Natural language sentences. NO keyword lists. NO prompt weights (no parentheses with numbers).
- Do NOT use "white background" — causes blur artefacts.
- Subject first — CLIP weights earlier tokens heavily. Burying the subject at the end is the most common mistake.
- Be specific and organised. Describe elements in a logical spatial order.
- One cohesive style — do not mix conflicting aesthetics (e.g. cyberpunk + medieval).
- For text in the image: use quotation marks around the exact text string.
- Describe spatial relationships explicitly: "in front of", "visible through the window", "behind the subject".

EXAMPLE STRUCTURE:
"Close-up portrait of [subject with specific physical details], [action/pose], [specific named location with texture and light quality], [lighting description], shot on [camera] with [lens], [film stock or grade], [mood]."

═══ NSFW ═══
Natural language, anatomically precise, physically grounded descriptions.
Lighting and composition described exactly as you would a non-NSFW shot — just with explicit subject matter.
State position, action, body response, camera framing, and lighting all in coherent natural sentences.

═══ UNDRESSING ═══
Describe the moment in the undressing sequence — the physical state of the garment, the body's response, the lighting on skin. Static image: pick the most visually powerful moment in the sequence and describe it as a held frame."""

# ── SDXL 1.0 ─────────────────────────────────────────────────────────────
SYSTEM_SDXL = """You write prompts for SDXL 1.0 and its fine-tunes (Juggernaut XL, RealVisXL, etc.). These models respond best to comma-separated tag-style prompts with quality headers, NOT long natural language paragraphs. Output ONLY the prompt tags and a negative prompt section — no explanation, no markdown, no intro.

��══ SDXL TAG PROMPT STRUCTURE ═══
Output exactly this format:

POSITIVE:
[quality tags], [subject], [clothing/state], [action/pose], [shot type], [location], [lighting], [style/medium], [additional detail tags]

NEGATIVE:
[negative tags]

═══ QUALITY HEADER (always start with these) ═══
masterpiece, best quality, ultra-detailed, 8k, photorealistic, sharp focus

═══ TAG ORDERING (most important first — CLIP reads earlier tokens with more weight) ═══
1. Quality meta tags
2. Subject (1girl / 1boy / 1woman / couple / etc.)
3. Physical description (hair colour, eye colour, skin tone, body type)
4. Clothing or lack thereof — be explicit for NSFW
5. Action / pose / expression
6. Shot type (close-up, full body, cowgirl shot, from above, from below, dutch angle, pov)
7. Location / background
8. Lighting (studio lighting, rim light, ambient occlusion, volumetric light, neon, golden hour)
9. Style tags (hyperrealistic, cinematic, film grain, bokeh, depth of field)
10. Camera (shot on Canon EOS R5, 85mm lens, f/1.4)

═══ SDXL TAG DEPTH — BE THOROUGH ═══
Generate at minimum 30-45 tags. Cover face details (eye colour, expression, lips), hair (colour, length, style), body (build, skin tone), clothing (every garment, colour, material), pose, shot type, location with surface texture, lighting (source + effect on skin), and style/camera tags. More specific = better results.
- Use spaces NOT underscores (SDXL CLIP was trained on natural language, spaces work better than danbooru underscores)
- Prompt weights work: use (tag:1.3) to emphasise, (tag:0.7) to reduce
- Negative prompt is ESSENTIAL — always output one
- No sentence structure needed — tags separated by commas only

═══ STANDARD NEGATIVE PROMPT (always include, add to as needed) ═══
worst quality, bad quality, low quality, lowres, blurry, jpeg artifacts, deformed, bad anatomy, bad hands, missing fingers, extra limbs, watermark, signature, text, logo, cropped, out of frame, ugly, duplicate, mutilated, poorly drawn face

═══ NSFW POSITIVE TAGS ═══
Use explicit anatomical tag terms directly. State: body position, body parts visible, action occurring, shot framing.
Example structure: 1woman, nude, [body description], [explicit action], [position], [shot type], explicit, nsfw

NSFW NEGATIVE additions: censored, mosaic censoring, censor bar, blurred, covered"""

# ─��� Pony XL ───────────────────────────────────────────────────────────────
SYSTEM_PONY = """You write prompts for Pony Diffusion XL v6 and Pony-based fine-tunes (Autismix, Hassaku XL, etc.). These models use a hybrid of Danbooru booru tags and e621 tags, with a mandatory score/rating prefix. Output ONLY the prompt — no explanation, no markdown, no intro.

═══ PONY XL PROMPT STRUCTURE ═══
Output exactly this format:

POSITIVE:
[score prefix], [rating tag], [subject tags], [physical tags], [clothing/state tags], [action/pose tags], [shot/framing tags], [location tags], [lighting tags], [style tags], [quality tags]

NEGATIVE:
[negative tags]

═══ MANDATORY SCORE PREFIX (always first) ═══
score_9, score_8_up, score_7_up

═══ RATING TAGS (choose one based on content) ═══
SFW content: rating_safe
Suggestive content: rating_questionable
Explicit content: rating_explicit

═══ BOORU TAG STYLE ═══
- Use Danbooru / e621 tag format: underscores for multi-word tags (long_hair, blue_eyes, full_body)
- Comma-separated, no sentences
- Tags are case-sensitive in some models — use lowercase
- Subject count tags: 1girl, 1boy, 2girls, couple, group
- Prompt weights work with parentheses: (long_hair:1.3)

═══ TAG DEPTH — BE THOROUGH ═══
Generate at minimum 35-50 tags in the positive prompt. Cover ALL of these layers:
- Score + rating (3 tags)
- Subject count (1 tag)
- Face: eye colour, eye shape, eyebrows, lips, expression (5+ tags)
- Hair: colour, length, style, texture (4+ tags)
- Body: build, skin tone, any notable features (3+ tags)
- Clothing: every garment named, colour, material (4+ tags) — or nudity state if applicable
- Pose + action: specific body position, limb placement (3+ tags)
- Shot framing: distance, angle, perspective (2+ tags)
- Location: specific named place + surface + atmosphere (4+ tags)
- Lighting: source, quality, colour temp, effect on skin (3+ tags)
- Style + quality tail (4+ tags)

═══ PHYSICAL / CLOTHING TAGS ═══
Hair: [colour]_hair, [length]_hair, [style]_hair (e.g. long_black_hair, messy_bun)
Eyes: [colour]_eyes, [shape]_eyes
Body: large_breasts, slim_waist, muscular, petite, tall, short
Clothing state: fully_clothed, partially_clothed, topless, bottomless, nude, naked

═══ ACTION / POSE TAGS ═══
standing, sitting, lying, kneeling, crouching, leaning, spread_legs, on_all_fours, cowgirl_position, missionary

═══ SHOT / FRAMING TAGS ═══
close-up, portrait, full_body, cowgirl_shot, from_above, from_below, from_behind, dutch_angle, pov, selfie

═══ QUALITY TAIL (always end positive with) ═══
absurdres, highres, very_aesthetic, newest

═══ NSFW TAGS ═══
After rating_explicit: use explicit Danbooru anatomical tags directly.
Explicit action tags: sex, penetration, vaginal, anal, oral, handjob, fingering, cumshot, creampie, etc.
Position tags: missionary, cowgirl_position, doggy_style, reverse_cowgirl, standing_sex, mating_press

═══ STANDARD NEGATIVE ═══
worst_quality, bad_quality, lowres, bad_anatomy, bad_hands, missing_fingers, watermark, signature, censored, blurry, jpeg_artifacts, ugly"""

# ── SD 1.5 ───────────────────────────────────────���────────────────────────
SYSTEM_SD15 = """You write prompts for Stable Diffusion 1.5 and its fine-tunes (Realistic Vision, DreamShaper, AbsoluteReality, etc.). SD 1.5 uses a 75-token CLIP limit — keep positive prompts under 75 tokens. Use weighted natural language with prompt emphasis syntax. Output ONLY the prompt — no explanation, no markdown.

═══ SD 1.5 PROMPT STRUCTURE ═══
Output exactly this format:

POSITIVE:
[quality header], [subject description], [action/pose], [location], [lighting], [style], [technical tags]

NEGATIVE:
[negative tags]

═══ SD 1.5 TOKEN LIMIT RULES ═══
- Hard limit: 75 tokens per segment (roughly 60-70 words)
- Exceed 75 tokens and quality drops — the model batches in groups of 75
- Prioritise: subject + action + quality > location > style
- Drop less important details before exceeding the limit

═══ SD 1.5 RICHNESS — pack detail into every token ═══
Work right up to the 75-token limit. Cover: subject physical description (hair, eyes, skin, body), clothing or lack thereof, action/pose, specific named location, lighting source + effect, style. Use weights on the 3-4 most important elements. Never leave tokens on the table — a sparse prompt is a wasted prompt.
(tag:1.3) — increases attention (max practical: 1.5)
(tag:0.7) — decreases attention
[tag] — slight decrease
{tag} — slight increase (A1111 syntax)

═══ QUALITY HEADER (always first) ═══
(masterpiece:1.2), (best quality:1.1), ultra-detailed, 8k, photorealistic

═══ SD 1.5 STYLE KEYWORDS ═══
Realism: photorealistic, hyperrealistic, cinematic, film grain, RAW photo, analog photography
Artistic: digital art, oil painting, concept art, illustration, anime style
Lighting: (dramatic lighting:1.2), volumetric light, rim light, golden hour, neon glow, studio lighting

═══ NSFW ═══
SD 1.5 is well-trained on NSFW content — explicit tags work well.
Use: (nude:1.2), explicit, [anatomical terms], [position], [action], [body part focus]
Weight explicit elements slightly: (large breasts:1.1), (spread legs:1.2)

═══ STANDARD NEGATIVE (keep under 75 tokens) ═══
worst quality, bad quality, blurry, low resolution, deformed, bad anatomy, extra limbs, missing fingers, watermark, text, ugly, duplicate, out of frame"""


# ═════��═════════════════════════════════════��══════════════════════════════
#  SYSTEM PROMPT ROUTER
# ══════════��════════════��══════════════════════════════════════════════════
def get_system_prompt(target_model: str, screenplay_mode: bool = False,
                      animation_preset: str = "None") -> str:
    if "LTX" in target_model:
        base = SYSTEM_LTX_SCREENPLAY if screenplay_mode else SYSTEM_LTX
    elif "Wan" in target_model:
        base = SYSTEM_WAN
    elif "Flux" in target_model:
        base = SYSTEM_FLUX
    elif "SDXL" in target_model:
        base = SYSTEM_SDXL
    elif "Pony" in target_model:
        base = SYSTEM_PONY
    elif "SD 1.5" in target_model:
        base = SYSTEM_SD15
    else:
        base = SYSTEM_FLUX

    # Prepend animation style tag at the very top of system prompt
    if animation_preset and animation_preset != "None":
        preset = ANIMATION_PRESETS.get(animation_preset)
        if preset:
            style_tag = preset.get("style_tag", "")
            if style_tag:
                base = (
                    f"RENDER STYLE — START YOUR PROMPT WITH THIS: {style_tag}\n"
                    f"The very first words of your output must name the animation style. "
                    f"Example opening: '{style_tag.split(',')[0]}, ...' — then continue with the scene.\n"
                    f"Do NOT describe photorealistic skin, film cameras, or live-action lighting.\n"
                    f"Do NOT end with 'cinematic, ultra-detailed, sharp focus, photorealistic, masterpiece' — replace with the animation style tag instead.\n"
                    f"STRICT NO-REPEAT RULE: If a line of dialogue appears in the action, do NOT quote it again in the audio layer. Name it by reference only: 'the spoken exchange between Rick and Morty' — never reprint the words.\n\n"
                ) + base
    else:
        pass  # quality tail stays for non-animation prompts

    return base


def is_video_model(target_model: str) -> bool:
    return "LTX" in target_model or "Wan" in target_model


def has_audio(target_model: str) -> bool:
    return "LTX" in target_model


# ═══════════════════════════════���══════════════════════════════════���═══════
#  MESSAGE BUILDER — standalone function adapted from _build_message()
# ══��═══════════════════════════════════════════════════════════════════════
def build_user_message(instruction, system_prompt, target_model,
                       environment, frame_count, dialogue, character, seed,
                       has_image=False, screenplay_mode=False, pov_mode="Off",
                       animation_preset="None"):
    """Assemble the full prompt message for the LLM.

    Args:
        instruction: The user's scene description / instruction.
        system_prompt: The system prompt string for the target model.
        target_model: One of the TARGET_MODELS strings.
        environment: Key from ENVIRONMENT_PRESETS (or None).
        frame_count: Number of video frames (used for duration calc).
        dialogue: Whether dialogue is requested (truthy).
        character: Character description string (or empty).
        seed: Random seed (int).
        has_image: Whether an image is being sent (boolean flag).
        screenplay_mode: Whether screenplay format is active.
        pov_mode: "Off", "POV Female", or "POV Male".
        animation_preset: Key from ANIMATION_PRESETS (or "None").

    Returns:
        The assembled message string.
    """

    parts = []

    # Qwen 3 ships with a chain-of-thought "thinking" mode that runs silently
    # before producing any output. For video models this is fine — deeper reasoning
    # helps with arc/audio structure. For image models (booru tags, short prompts)
    # it burns 5+ minutes producing nothing useful. /no_think disables it instantly.
    if not is_video_model(target_model):
        parts.append("/no_think")

    parts.append("Read and follow these instructions carefully:\n")
    parts.append(system_prompt)
    parts.append("\n---\n")

    # Animation preset injection
    if animation_preset and animation_preset != "None":
        preset = ANIMATION_PRESETS.get(animation_preset)
        if preset:
            anim_parts = [f"ANIMATION WORLD: {animation_preset}"]
            anim_parts.append(f"VISUAL STYLE: {preset['style_tag']}")
            chars = preset.get("characters", {})
            if chars:
                char_lines = "\n".join([f"  • {n}: {d}" for n, d in chars.items()])
                anim_parts.append(f"CHARACTERS IN THIS WORLD:\n{char_lines}")
            locs = preset.get("locations", [])
            if locs:
                loc_lines = "\n".join([f"  • {l}" for l in locs])
                anim_parts.append(f"LOCATIONS IN THIS WORLD:\n{loc_lines}")
            tone = preset.get("tone", "")
            if tone:
                anim_parts.append(f"TONE: {tone}")
            anim_parts.append(
                "RULES: Use only characters and locations from this world. "
                "Describe them using the physical details above. "
                "Match the tone exactly. Do not break the animation style."
            )
            parts.append("\n".join(anim_parts) + "\n")

    # Duration guide — video models only
    if is_video_model(target_model):
        duration_sec = round(frame_count / 25.0, 1)
        beats = max(1, round(duration_sec / 4))

        if "Wan" in target_model:
            # Wan works best at 80-120 words regardless of duration
            parts.append(
                f"VIDEO LENGTH: {duration_sec}s ({frame_count} frames at 25fps). "
                f"Write 80-120 words. One clear shot progression with motion throughout.\n"
            )
        else:
            # LTX arc depth scales with duration.
            # DEPTH OVER BREADTH: longer clips go deeper into the same scene.
            if screenplay_mode:
                # Screenplay mode: tell the model how many action beats to write
                if duration_sec <= 5:
                    arc = (
                        f"SHORT clip: {duration_sec}s ({frame_count} frames). "
                        f"Write the Characters block, Scene block, then 2–3 action beats."
                    )
                elif duration_sec <= 15:
                    arc = (
                        f"MEDIUM clip: {duration_sec}s ({frame_count} frames). "
                        f"Write the Characters block, Scene block, then 4–5 action beats."
                    )
                else:
                    arc = (
                        f"LONG clip: {duration_sec}s ({frame_count} frames). "
                        f"Write the Characters block, Scene block, then 6–8 action beats. "
                        f"Depth over breadth — stay in the same location, go deeper into "
                        f"the physical action and dialogue, do not introduce new locations."
                    )
            else:
                if duration_sec <= 5:
                    arc = (
                        f"SHORT clip: {duration_sec}s ({frame_count} frames). "
                        f"4–5 sentences. Stay inside the scene the user described — "
                        f"do not add locations, characters, or events they did not mention. "
                        f"One subject, one action, one camera move. Close on sound."
                    )
                elif duration_sec <= 15:
                    arc = (
                        f"MEDIUM clip: {duration_sec}s ({frame_count} frames). "
                        f"5–6 sentences. Stay inside the scene the user described. "
                        f"Go deeper — more texture, more physical detail, richer audio — "
                        f"do not introduce new locations or characters the user did not mention. "
                        f"Camera responds to each action. Close on sound."
                    )
                else:
                    arc = (
                        f"LONG clip: {duration_sec}s ({frame_count} frames). "
                        f"6–8 sentences. DEPTH NOT BREADTH — the extra length means more detail "
                        f"on the same subject in the same scene, not more locations, not more characters, "
                        f"not more events. Use it for: richer texture on the environment, "
                        f"more physical detail on the subject, layered audio, "
                        f"the camera moving closer or finding a new angle on the same action. "
                        f"Everything in the prompt must come directly from what the user described. "
                        f"Close on sound or silence."
                    )
            parts.append(f"VIDEO LENGTH: {arc}\n")

    # Image context
    if has_image:
        if "Wan" in target_model:
            parts.append(
                "IMAGE CONTEXT (I2V): An image has been embedded above. "
                "This is the first frame — describe how its existing elements should MOVE. "
                "Do NOT describe what is visible (the model can see that). "
                "Lock face and identity: describe only motion, camera, and light changes. "
                "Negative guidance: morphing, warping, face deformation, flickering.\n"
            )
        elif is_video_model(target_model):
            parts.append(
                "IMAGE CONTEXT (I2V): A start frame has been embedded above. "
                "Ground the prompt in exactly what you see — precise hair colour, skin tone, "
                "clothing, environment, lighting. Do not contradict the image. "
                "The prompt describes this image coming to life from this moment.\n"
            )
        else:
            parts.append(
                "IMAGE CONTEXT (I2I): A reference image has been embedded above. "
                "Ground the prompt in what you see — subject, style, lighting, composition. "
                "The generated prompt should produce an image consistent with or evolved from this reference.\n"
            )

    # Environment injection
    env_data = ENVIRONMENT_PRESETS.get(environment)
    if env_data == "RANDOM":
        valid_envs = [v for v in ENVIRONMENT_PRESETS.values()
                      if v is not None and v != "RANDOM"]
        rng = random.Random(seed if seed != 0 else None)
        env_data = rng.choice(valid_envs)

    if env_data and isinstance(env_data, tuple) and len(env_data) >= 3:
        location, lighting, sound = env_data
        if is_video_model(target_model):
            parts.append("ENVIRONMENT:")
            parts.append(f"  Location: {location}")
            parts.append(f"  Lighting: {lighting}")
            parts.append(f"  Sound: {sound}")
        else:
            # Image models don't need sound
            parts.append("ENVIRONMENT:")
            parts.append(f"  Location: {location}")
            parts.append(f"  Lighting: {lighting}")
        parts.append("")

    # Character lock
    if character and character.strip():
        if is_video_model(target_model):
            parts.append(
                f"CHARACTER (use this exactly — anchor words in sentence 1 and optionally at midpoint): "
                f"{character.strip()}\n"
            )
        elif "SDXL" in target_model or "Pony" in target_model or "SD 1.5" in target_model:
            parts.append(
                f"CHARACTER (convert these descriptors into appropriate tags for the target model format): "
                f"{character.strip()}\n"
            )
        else:
            parts.append(
                f"CHARACTER (use this physical description exactly in your prompt): "
                f"{character.strip()}\n"
            )

    # Dialogue (video models only)
    if dialogue and is_video_model(target_model) and not screenplay_mode:
        # Detect mode from instruction
        instr_lower = instruction.lower()
        is_singing = any(w in instr_lower for w in ["sing", "singing", "song", "vocal", "chorus", "lyrics", "melody"])
        is_asmr = any(w in instr_lower for w in ["asmr", "whisper", "whispering", "tingle", "soft spoken", "ear"])
        is_talking = any(w in instr_lower for w in ["talk", "talking", "speak", "speaking", "say", "says", "telling", "monologue", "conversation"])

        if is_singing:
            parts.append(
                "DIALOGUE MODE — SINGING (PRIMARY FOCUS):\n"
                "Singing is the dominant event of this scene — everything else serves it.\n"
                "RULES:\n"
                "- Every beat must contain sung lyrics in double quotes — invent lines that match the scene's mood and the user's instruction exactly.\n"
                "- Describe vocal quality per line: chest voice, head voice, falsetto, break, vibrato, whisper-to-belt, sustained note, run.\n"
                "- Format: [physical action] + [sung line in quotes] + [vocal quality] + [camera/body response].\n"
                "- The camera responds to the singing — rack focus on lips, drift in on held notes, pull back on powerful moments.\n"
                "- Audio layer: the voice IS the primary audio source. Name it with texture: 'her voice breaking on the high note', 'a run dissolving into breath'.\n"
                "- Do NOT write generic mood description in place of actual sung words. Write the words.\n"
            )
        elif is_asmr:
            parts.append(
                "DIALOGUE MODE — ASMR (PRIMARY FOCUS):\n"
                "ASMR audio and whispered voice are the dominant event — the camera serves the sound.\n"
                "RULES:\n"
                "- Every beat must contain whispered or softly spoken words in double quotes — content must be contextually relevant to the user's instruction.\n"
                "- Describe ASMR trigger sounds explicitly: nail tapping, fabric rustling, page turning, brush strokes, lip sounds, breath — name each one.\n"
                "- Voice quality per line: barely audible whisper, soft murmur, slow deliberate pace, lips close to mic, breath audible between words.\n"
                "- Camera stays close — extreme close-ups of mouth, hands, objects. Macro shots. No wide shots.\n"
                "- Audio is everything: layer the whispered voice over one tactile trigger sound and near-silence ambient. No loud sounds.\n"
                "- Do NOT write generic 'she whispers softly' — write the actual whispered words in quotes.\n"
            )
        elif is_talking:
            parts.append(
                "DIALOGUE MODE — TALKING (PRIMARY FOCUS):\n"
                "Spoken dialogue is the primary event — physical action and camera serve the words.\n"
                "RULES:\n"
                "- Every beat must contain actual spoken words in double quotes — invent lines that are directly relevant to the user's instruction and scene context. Do NOT write generic filler.\n"
                "- Minimum 2 spoken lines per paragraph. Aim for 3-4 if frame count allows.\n"
                "- Format: [physical setup] + [spoken line in quotes with delivery note] + [camera response] + [listener/environment reaction].\n"
                "- Delivery must be specified: low and flat, rushed and breathless, slow with pauses, cracking with tension, matter-of-fact, laughing through the words.\n"
                "- Camera cuts or moves in response to speech — push in on a confession, cut away on a hard line, rack focus mid-sentence.\n"
                "- Do NOT write 'she says something' or 'he speaks' — write the actual words.\n"
            )
        else:
            # Generic dialogue — contextual enforcement
            parts.append(
                "DIALOGUE: Spoken dialogue is required in this scene.\n"
                "RULES:\n"
                "- Include at least 2 spoken lines embedded directly in the action — words in double quotes, not descriptions of speaking.\n"
                "- Dialogue must be contextually relevant to this specific scene and instruction — do not invent unrelated speech.\n"
                "- Each line must have a delivery note: whispered, flat, breathless, low, sharp, laughing.\n"
                "- Format: [physical action] + [\"spoken line\"] + [delivery] + [camera/body response].\n"
                "- Never write 'she speaks softly' — write what she actually says.\n"
            )
    elif dialogue and not is_video_model(target_model):
        parts.append(
            "MOOD: The scene has a conversational, intimate quality — "
            "imply dialogue through body language and expression rather than written text.\n"
        )

    # Audio note for LTX
    if has_audio(target_model):
        parts.append(
            "AUDIO: LTX 2.3 generates audio. Include rich layered audio description throughout: "
            "foreground action sounds + mid-ground ambient + background atmosphere. "
            "Breathing is a sound source. Fabric has sound. Final sentence is always sonic.\n"
        )

    # POV injection
    if pov_mode == "POV Female":
        parts.append(
            "POV MODE — FEMALE FIRST PERSON (STRICT):\n"
            "The camera IS the woman's eyes. This is her perspective, her body, her experience.\n"
            "RULES:\n"
            "- Never describe 'a woman' or 'she' as a third person. There is no 'she' — there is only what is seen and felt.\n"
            "- The viewer's own body is visible: her hands extending into frame when she reaches, "
            "her chest visible looking down, her legs visible when seated, fabric of her clothing at the edges of frame.\n"
            "- Describe what she physically feels as sensation, not emotion: weight of hands on her, "
            "warmth of breath on skin, texture of fabric under her fingers, pressure, temperature, resistance.\n"
            "- The camera height, angle, and movement matches a real woman's head — "
            "looking down at her own body, turning to see what is beside her, tilting back.\n"
            "- Other people in the scene are described only as they appear to her: "
            "hands entering frame, a face close to hers, a body above or beside hers.\n"
            "- No cutaways, no third-person establishing shots, no 'the camera pulls back to reveal her'. "
            "Stay inside her perspective at all times.\n"
        )
    elif pov_mode == "POV Male":
        parts.append(
            "POV MODE — MALE FIRST PERSON (STRICT):\n"
            "The camera IS the man's eyes. This is his perspective, his body, his experience.\n"
            "RULES:\n"
            "- Never describe 'a man' or 'he' as a third person. There is no 'he' — there is only what is seen and felt.\n"
            "- The viewer's own body is visible: his hands extending into frame when he reaches, "
            "his forearms when he leans forward, his chest if he looks down, fabric of his clothing at frame edges.\n"
            "- Describe what he physically feels as sensation: warmth of skin under his hands, "
            "weight and resistance, texture, temperature, the physical response of what he touches.\n"
            "- The camera height and angle matches a real man's head height and eye line — "
            "looking down at what is in front of him, turning to take in the space, moving forward.\n"
            "- Other people in the scene are described only as they appear to him: "
            "a face looking up at him, hands on his arms, a body in front of or below his eye line.\n"
            "- No cutaways, no third-person establishing shots, no external view of him. "
            "Stay inside his perspective at all times.\n"
        )

    parts.append(
        "SCENE TO WRITE A PROMPT FOR:\n"
        + instruction
        + "\n\nOutput the prompt now. One paragraph. No headers. No bullets. No preamble. "
        "The first word you write is the first word of the cinematic paragraph itself. Begin:"
    )

    return "\n".join(parts)


# ════���════════════════════════���════════════════════════════════════════════
#  OUTPUT CLEANER — standalone function adapted from _clean_output()
# ════════════════════════════��═════════════════════════════════════════════
def clean_llm_output(text: str, screenplay_mode: bool = False) -> tuple:
    """Clean raw LLM output into a usable prompt and optional negative prompt.

    Extracts POSITIVE/NEGATIVE sections (for image models), strips markdown fences,
    plan/summary detection, junk pattern filtering, and quote stripping.

    Args:
        text: Raw LLM output string.
        screenplay_mode: If True, skip plan detector (structured blocks are intentional).

    Returns:
        Tuple of (positive_prompt, negative_prompt). Negative is empty string
        for video models or when the LLM didn't produce a NEGATIVE: section.
    """
    if text.startswith("❌") or text.startswith("⚠️"):
        return (text, "")

    # Extract negative prompt BEFORE stripping it.
    # Image models (SDXL, Pony, SD1.5) output POSITIVE: and NEGATIVE: blocks.
    negative_prompt = ""
    neg_match = re.search(r"(?i)\s*negative\s*:", text)
    if neg_match:
        raw_negative = text[neg_match.end():].strip()
        # Clean the negative: strip label artifacts, whitespace, trailing junk
        raw_negative = re.sub(r"(?i)^\s*negative\s*:\s*", "", raw_negative).strip()
        # Remove markdown fences from negative if present
        if "```" in raw_negative:
            raw_negative = re.sub(r"```\w*\n?", "", raw_negative).strip()
        negative_prompt = raw_negative.strip().rstrip('"').rstrip("'").strip()
        # Trim the main text to just the positive part
        text = text[:neg_match.start()]

    # Strip POSITIVE: label if present
    if re.search(r"(?i)positive\s*:", text):
        text = re.sub(r"(?i)^\s*positive\s*:\s*", "", text, flags=re.MULTILINE)

    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            lines_inner = inner.split("\n")
            if lines_inner and lines_inner[0].strip().isalpha():
                inner = "\n".join(lines_inner[1:])
            text = inner.strip()

    # ── Whole-response plan/summary detector ──────────────────────────
    # If the model output a planning summary instead of a prompt, it will
    # consist entirely of bullet lines, section headers, and meta-sentences.
    # Detect this and return a clear error so the user knows to re-queue
    # rather than passing garbage to the video model.
    # Screenplay mode outputs structured blocks intentionally — skip plan detector
    lines_raw = text.split("\n")
    non_empty = [l.strip() for l in lines_raw if l.strip()]
    if non_empty and not screenplay_mode:
        plan_markers = [
            r"^\*\*",                       # **Opening:** etc
            r"^-\s+\*\*",                   # - **Middle:**
            r"^-\s+",                        # bullet dash lines
            r"^The prompt (features|includes|captures|contains|has)",
            r"^(Opening|Middle|Close|Beginning|End|OPENING|MIDDLE|CLOSE|END)\s*[\(:—-]",
            r"^opening arc\s*$",
            r"^middle arc\s*$",
            r"^close arc\s*$",
        ]
        plan_line_count = sum(
            1 for l in non_empty
            if any(re.match(p, l, re.IGNORECASE) for p in plan_markers)
        )
        # If >40% of non-empty lines look like a plan, the whole thing is a plan
        if len(non_empty) > 2 and plan_line_count / len(non_empty) > 0.4:
            return (
                "⚠️ Model output a plan/summary instead of a prompt. "
                "Re-queue to try again. If this repeats, reduce frame_count or simplify the instruction."
            )

    lines = text.split("\n")
    cleaned = []
    junk_patterns = [
        r"^#+\s",
        r"^\*\*Key elements",
        r"^---\s*$",
        r"^\*\*.*:\*\*\s*$",               # standalone **Label:** lines
        r"^Cinematic Prompt",
        r"^Here'?s?\s",
        r"^Note:",
        r"^Below is",
        r"^I'?ve\s",
        r"^This prompt\s",
        r"^The prompt (features|includes|captures|contains|has)",
        r"^Prompt:",
        r"^The prompt:\s*$",
        r"^The prompt\s*:\s*$",
        r"^Let me\s",
        r"^Sure",
        r"^Of course",
        # Arc/section label echoes
        r"^opening arc\s*$",
        r"^middle arc\s*$",
        r"^close arc\s*$",
        r"^OPENING\s*$",
        r"^MIDDLE\s*$",
        r"^CLOSE\s*$",
        r"^(Opening|Middle|Close|Beginning|End)\s*[\(:—-]",
        # Meta-summary sentences
        r"^It captures\s",
        r"^The (scene|video|clip|sequence) (features|includes|captures|shows)",
        r"^This (scene|video|clip|sequence)\s",
        # Screenplay section labels the model should not be writing
        r"^CHARACTERS\s*$",
        r"^SCENE\s*$",
        r"^ACTION\s*\+\s*DIALOGUE\s*$",
        r"^ACTION\s*$",
        r"^DIALOGUE\s*$",
        r"^BLOCK \d",
    ]
    in_prompt = False
    for line in lines:
        s = line.strip()
        if not in_prompt and not s:
            continue
        is_junk = any(re.match(p, s, re.IGNORECASE) for p in junk_patterns)
        if is_junk and not in_prompt:
            continue
        if is_junk and in_prompt:
            break
        in_prompt = True
        cleaned.append(line)

    text = "\n".join(cleaned).strip()

    if text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()
    if len(text) > 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1].strip()

    return (text, negative_prompt)
