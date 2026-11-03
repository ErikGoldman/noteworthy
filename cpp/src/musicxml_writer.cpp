#include "musicxml_writer.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <sstream>

namespace omr {
namespace {

const char* type_name(int base) {
  switch (base) {
    case 1: return "whole";
    case 2: return "half";
    case 4: return "quarter";
    case 8: return "eighth";
    case 16: return "16th";
    case 32: return "32nd";
    case 64: return "64th";
  }
  return "quarter";
}

const char* clef_sign(ClefSign s) {
  switch (s) {
    case ClefSign::G: return "G";
    case ClefSign::F: return "F";
    case ClefSign::C: return "C";
  }
  return "G";
}

std::string escape_xml(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      default: out += c;
    }
  }
  return out;
}

// Smallest divisions-per-quarter that represents every onset and duration exactly.
int64_t divisions_for(const Score& score) {
  int64_t div = 1;
  for (const Measure& m : score.measures) {
    for (const Event& e : m.events) {
      div = std::lcm(div, e.onset.den());
      div = std::lcm(div, e.duration.den());
    }
  }
  return div;
}

class MeasureWriter {
 public:
  MeasureWriter(std::ostringstream& out, int64_t divisions) : out_(out), divisions_(divisions) {}

  void write_events(const Measure& measure) {
    std::map<std::pair<int, int>, std::vector<const Event*>> voices;  // (staff, voice) -> events
    for (const Event& e : measure.events) voices[{e.staff, e.voice}].push_back(&e);
    bool first = true;
    for (auto& [key, events] : voices) {
      if (!first) backup_to_start();
      first = false;
      std::sort(events.begin(), events.end(), [](const Event* a, const Event* b) { return a->onset < b->onset; });
      for (const Event* e : events) write_event(*e);
    }
  }

 private:
  int64_t ticks(const Rational& r) const { return (r * Rational(divisions_)).num(); }

  void backup_to_start() {
    if (cursor_ > Rational(0)) out_ << "<backup><duration>" << ticks(cursor_) << "</duration></backup>";
    cursor_ = Rational(0);
  }

  void write_event(const Event& e) {
    if (e.onset > cursor_) {
      out_ << "<forward><duration>" << ticks(e.onset - cursor_) << "</duration><voice>" << e.voice
           << "</voice><staff>" << e.staff << "</staff></forward>";
    } else if (e.onset < cursor_) {
      // Overlapping events within one voice cannot be expressed sequentially; back up.
      out_ << "<backup><duration>" << ticks(cursor_ - e.onset) << "</duration></backup>";
    }
    cursor_ = e.onset + e.duration;

    if (e.is_rest()) {
      out_ << "<note><rest/>";
      write_note_fields(e, /*tie_start=*/false);
      out_ << "</note>";
      return;
    }
    for (size_t i = 0; i < e.pitches.size(); ++i) {
      const Pitch& p = e.pitches[i];
      const bool tie = i < e.tie_start.size() && e.tie_start[i];
      out_ << "<note>" << (i ? "<chord/>" : "") << "<pitch><step>" << p.step_name() << "</step>";
      if (p.alter) out_ << "<alter>" << p.alter << "</alter>";
      out_ << "<octave>" << p.octave() << "</octave></pitch>";
      write_note_fields(e, tie);
      if (tie) out_ << "<notations><tied type=\"start\"/></notations>";
      out_ << "</note>";
    }
  }

  // Everything after <pitch>/<rest>, in the element order the MusicXML schema requires.
  void write_note_fields(const Event& e, bool tie_start) {
    out_ << "<duration>" << ticks(e.duration) << "</duration>";
    if (tie_start) out_ << "<tie type=\"start\"/>";
    out_ << "<voice>" << e.voice << "</voice><type>" << type_name(e.value.base) << "</type>";
    for (int d = 0; d < e.value.dots; ++d) out_ << "<dot/>";
    if (e.value.in_tuplet()) {
      out_ << "<time-modification><actual-notes>" << e.value.tuplet_actual << "</actual-notes><normal-notes>"
           << e.value.tuplet_normal << "</normal-notes></time-modification>";
    }
    out_ << "<staff>" << e.staff << "</staff>";
  }

  std::ostringstream& out_;
  int64_t divisions_;
  Rational cursor_{0};
};

void write_attributes(std::ostringstream& out, const Measure& m, int64_t divisions, bool first, int staves) {
  if (!first && !m.key_fifths && !m.time && m.clefs.empty()) return;
  out << "<attributes>";
  if (first) out << "<divisions>" << divisions << "</divisions>";
  if (m.key_fifths) out << "<key><fifths>" << *m.key_fifths << "</fifths></key>";
  if (m.time) out << "<time><beats>" << m.time->beats << "</beats><beat-type>" << m.time->beat_type << "</beat-type></time>";
  if (first) out << "<staves>" << staves << "</staves>";
  for (const auto& [staff, clef] : m.clefs) {
    out << "<clef number=\"" << staff << "\"><sign>" << clef_sign(clef.sign) << "</sign><line>" << clef.line
        << "</line></clef>";
  }
  out << "</attributes>";
}

}  // namespace

std::string to_musicxml(const Score& score, const std::string& title) {
  const int64_t divisions = divisions_for(score);
  std::ostringstream out;
  out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
      << "<score-partwise version=\"3.1\"><work><work-title>" << escape_xml(title) << "</work-title></work>"
      << "<part-list><score-part id=\"P1\"><part-name>Piano</part-name></score-part></part-list><part id=\"P1\">";
  for (size_t i = 0; i < score.measures.size(); ++i) {
    const Measure& m = score.measures[i];
    out << "<measure number=\"" << i << "\">";
    write_attributes(out, m, divisions, i == 0, score.staves);
    MeasureWriter(out, divisions).write_events(m);
    out << "</measure>";
  }
  out << "</part></score-partwise>\n";
  return out.str();
}

}  // namespace omr
