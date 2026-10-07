// The drum type a drum patch's name suggests, as the short label of a
// channel button (BD, SD, CH, OH, CP, RS, LT, MT, HT, CB, ...), or '' when
// the name gives none (the button then shows the channel number). Pure.
//
// Abbreviations match as words (digits may touch them: "808BD"); longer
// words and phrases are matched anywhere in the name.

const word = (abbr) => new RegExp(`(?<![A-Z])(?:${abbr})(?![A-Z])`, 'i');

// First match wins: the specific names come before the general ones
const RULES = [
  ['RS', word('RS|RIM') , /RIM\s*SHOT|SIDE\s*STICK/i],
  ['CP', word('CP|CLP|HC'), /CLAP/i],
  ['LT', word('LT'), /(LOW|LO)\s*TOM|TOM\s*(LOW|LO)(?![A-Z])/i],
  ['MT', word('MT'), /MID\s*TOM|TOM\s*MID/i],
  ['HT', word('HT'), /(HIGH|HI)\s*TOM|TOM\s*(HIGH|HI)(?![A-Z])/i],
  ['TOM', word('TOM|TOMS'), null],
  ['BD', word('BD|BDRUM'), /KICK|BASS\s*DR/i],
  ['SD', word('SD|SNR'), /SNARE/i],
  ['OH', word('OH|OHH'), /OPEN\s*H|OP\s*HAT/i],
  ['CH', word('CH|CHH'), /CLOSED\s*H|CL\s*HAT/i],
  ['HH', word('HH|HAT'), /HI\s*-?\s*HAT/i],
  ['CB', word('CB'), /COW\s*BELL/i],
  ['RC', word('RC'), /RIDE/i],
  ['CC', word('CC'), /CRASH/i],
  ['CY', word('CY|CYM'), /CYMBAL/i],
  ['CG', word('CG'), /CONGA|BONGO/i],
  ['TB', word('TB|TAMB'), /TAMBOURINE/i],
  ['SH', word('SH'), /SHAKER|CABASA|MARACA/i],
  ['CL', word('CL'), /CLAVE/i],
  ['PC', word('PERC'), /PERCUSSION/i],
  ['FX', word('FX|ZAP|BLIP'), /NOISE|SYNTH|REVERSE|FUZZ/i],
];

/** The short drum type label of a drum patch name, or ''. */
export function guessDrumType(name) {
  if (!name) return '';
  for (const [label, abbr, phrase] of RULES) {
    if (abbr.test(name) || (phrase && phrase.test(name))) return label;
  }
  return '';
}
