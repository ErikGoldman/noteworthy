#include "score.h"

namespace omr {
namespace {

constexpr int kStepSemitones[7] = {0, 2, 4, 5, 7, 9, 11};
constexpr char kStepNames[7] = {'C', 'D', 'E', 'F', 'G', 'A', 'B'};

int floor_div(int a, int b) { return a / b - ((a % b != 0) && ((a < 0) != (b < 0))); }
int floor_mod(int a, int b) { return a - floor_div(a, b) * b; }

}  // namespace

int Clef::bottom_line_step() const {
  // Step of the clef's own reference pitch, minus the distance from the bottom line.
  int reference = 0;
  switch (sign) {
    case ClefSign::G: reference = 4; break;   // G4
    case ClefSign::F: reference = -4; break;  // F3
    case ClefSign::C: reference = 0; break;   // C4
  }
  return reference - 2 * (line - 1);
}

char Pitch::step_name() const { return kStepNames[floor_mod(diatonic, 7)]; }
int Pitch::octave() const { return 4 + floor_div(diatonic, 7); }
int Pitch::midi() const { return 12 * (octave() + 1) + kStepSemitones[floor_mod(diatonic, 7)] + alter; }

Rational NoteValue::duration() const {
  Rational d(4, base);
  Rational add = d;
  for (int i = 0; i < dots; ++i) {
    add = add / 2;
    d += add;
  }
  return d;
}

}  // namespace omr
