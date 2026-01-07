\version "2.24.0"
\header { title = "Iteration 19" }

music = {
  \tempo 4 = 117
  a,16\5 r1 e16\4 r1 g8\3 r1 a,8\5 r2
  a,8\5 r4 c16\5 r2 e4\4 e4\4 g'8\1 r1
  e8\4 r2 a,8\5 r4 f16\4 r1 a,8\5 r2
  a,16\5 r4 a,16\5 r1 g4\3 g8\3 g4\3 a,8\5
  d2\4 r4 d16\4 r2 c8\5 r1 c4\5 e4\4
  r1 d8\4 r1 c16\5 g8\3 c16\5 r4 g4\3
  <c\5 g\3>4 d4\4 r1 c4\5 r1 c4\5 f8\4 c8\5
  r1 f4\4 c4\5 r4 g8\3 r1 a,16\5 r2
  e8\4 d8\4 r2 a,8\5 r2 a,8\5
}

\score {
  <<
    \new Staff { \clef "treble" \music }
    \new TabStaff { \clef "moderntab" \tabFullNotation \music }
  >>
  \layout { }
  \midi { }
}
