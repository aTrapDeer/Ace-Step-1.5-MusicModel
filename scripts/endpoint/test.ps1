param(
  [string]$Token = "",
  [string]$Url   = "",
  [string]$Prompt = "upbeat pop rap with emotional guitar",
  [string]$Lyrics = "",
  [int]$DurationSec = 3,
  [int]$SampleRate  = 44100,
  [int]$Seed        = 42,
  [double]$GuidanceScale = 7.0,
  [int]$Steps = 50,
  [string]$UseLM = "true",
  [switch]$SimplePrompt,
  [switch]$Instrumental,
  [switch]$RnbLoveTemptation,
  [switch]$RnbPopRap2Min,
  [switch]$AllowFallback,
  [string]$OutFile  = "test_music.wav"
)

$ErrorActionPreference = "Stop"

if (-not $Token) {
  $Token = $env:HF_TOKEN
}
if (-not $Url) {
  $Url = $env:HF_ENDPOINT_URL
}

if (-not $Token) {
  throw "HF token not provided. Use -Token or set HF_TOKEN."
}
if (-not $Url) {
  throw "Endpoint URL not provided. Use -Url or set HF_ENDPOINT_URL."
}

if ($RnbLoveTemptation.IsPresent) {
  $Prompt = "melodic RnB pop with ambient melodies, evolving chord progression, emotional modern production, intimate vocals, soulful hooks"
  $Lyrics = @"
[Verse 1]
Late night shadows on my skin,
I hear your name where the silence has been.
I swore I'd run when the fire got close,
But I keep chasing what hurts me the most.

[Pre-Chorus]
Every promise pulls me back again,
Sweet poison dressed like a loyal friend.

[Chorus]
I'm fighting love and temptation,
Heart in a war with my own salvation.
One touch and my defenses break,
I know it's danger but I stay awake.
I'm fighting love and temptation,
Drowning slow in this sweet devastation.
I want to leave but I hesitate,
Cause I still crave what I know I should hate.

[Verse 2]
Your voice is velvet over broken glass,
I learn the pain but I still relapse.
Truth on my lips, lies in my veins,
I pray for peace while I dance in flames.

[Bridge]
If love is a test, I'm failing with grace,
Still falling for fire, still calling your name.

[Final Chorus]
I'm fighting love and temptation,
Heart in a war with my own salvation.
One touch and my defenses break,
I know it's danger but I stay awake.
"@

  if ($DurationSec -eq 3) {
    $DurationSec = 24
  }

  if (-not $PSBoundParameters.ContainsKey("SimplePrompt")) {
    $SimplePrompt = $false
  }
}

if ($RnbPopRap2Min.IsPresent) {
  $Prompt = "2 minute RnB Pop Rap song, melodic ambient pads, emotional chord progression, intimate female and male vocal blend, catchy hooks, modern drums, deep 808, vulnerable but confident tone"
  $Lyrics = @"
[Intro]
Mm, midnight in my head, no sleep.
Same old war in my chest, still deep.
I say I'm done, then I call your name,
I run from the fire, then I walk in the flame.

[Verse 1]
Streetlights drip on the window pane,
I wear my pride like a silver chain.
Say I don't need you, that's what I say,
But your voice in my mind don't fade away.
I got dreams, got scars, got bills to pay,
Still I fold when your eyes pull me in that way.
I know better, I swear I do,
But temptation sounds like truth when it sounds like you.

[Pre-Chorus]
Every promise tastes sweet then turns to smoke,
I keep rebuilding hearts that we already broke.
I want peace, but I want your touch,
I know it's too much, still it's never enough.

[Chorus]
I'm fighting love and temptation,
Heart on trial with no salvation.
One more kiss and the walls cave in,
I lose myself just to feel again.
I'm fighting love and temptation,
Drowning slow in sweet devastation.
I say goodbye, then I hesitate,
Cause I still crave what I know I should hate.

[Rap Verse 1]
Look, I been in and out the same lane,
Different night, same rain.
Tell myself "don't text back,"
Still type your name, press send, same pain.
You the high and the low in one dose,
Got me praying for distance, still close.
I play tough, but the truth is loud,
When you're gone, all this noise in the crowd.
Yeah, I hustle, I grind, I glow,
But alone in the dark, I'm a different soul.
If love was logic, I'd be free by now,
But my heart ain't science, I just bleed it out.

[Verse 2]
Your perfume still lives in my hoodie seams,
Like a ghost in the corners of all my dreams.
I learned your chaos, your every disguise,
The saint in your smile, the storm in your eyes.
I touch your hand and forget my name,
Call it desire, call it blame.
I need healing, I need release,
But your lips keep turning my war to peace.

[Pre-Chorus]
Every promise tastes sweet then turns to smoke,
I keep rebuilding hearts that we already broke.
I want peace, but I want your touch,
I know it's too much, still it's never enough.

[Chorus]
I'm fighting love and temptation,
Heart on trial with no salvation.
One more kiss and the walls cave in,
I lose myself just to feel again.
I'm fighting love and temptation,
Drowning slow in sweet devastation.
I say goodbye, then I hesitate,
Cause I still crave what I know I should hate.

[Rap Verse 2]
Uh, late calls, no sleep, red eyes,
Truth hurts more than sweet lies.
We toxic, but the chemistry loud,
Like thunder in a summer night over this town.
Tell me leave, then you pull me near,
Tell me "trust me," then feed my fear.
I keep faith in a broken map,
Tryna find us on roads that don't lead back.
I got plans, got goals, got pride,
But temptation got hands on the wheel tonight.
If I fall, let me fall with grace,
I still see home when I look in your face.

[Bridge]
If this love is a test, I'm failing in style,
Smiling through fire for one more while.
I know I should run, I know I should wait,
But your name on my tongue sounds too much like fate.

[Final Chorus]
I'm fighting love and temptation,
Heart on trial with no salvation.
One more kiss and the walls cave in,
I lose myself just to feel again.
I'm fighting love and temptation,
Drowning slow in sweet devastation.
I say goodbye, then I hesitate,
Cause I still crave what I know I should hate.

[Outro]
Mm, midnight in my head, no sleep.
Still your name in my chest, too deep.
"@

  if ($DurationSec -eq 3) {
    $DurationSec = 120
  }

  if (-not $PSBoundParameters.ContainsKey("SimplePrompt")) {
    $SimplePrompt = $false
  }
}

$useLmBool = $true
if ($null -ne $UseLM -and $UseLM -ne "") {
  try {
    $useLmBool = [System.Convert]::ToBoolean($UseLM)
  }
  catch {
    $useLmBool = ($UseLM -match '^(1|true|t|yes|y|on)$')
  }
}

$inputs = @{
  prompt         = $Prompt
  duration_sec   = $DurationSec
  sample_rate    = $SampleRate
  seed           = $Seed
  guidance_scale = $GuidanceScale
  steps          = $Steps
  use_lm         = $useLmBool
  allow_fallback = $AllowFallback.IsPresent
}

if ($Lyrics) {
  $inputs["lyrics"] = $Lyrics
}
if ($SimplePrompt.IsPresent) {
  $inputs["simple_prompt"] = $true
}
if ($Instrumental.IsPresent) {
  $inputs["instrumental"] = $true
}

$body = @{ inputs = $inputs } | ConvertTo-Json -Depth 8

$response = Invoke-RestMethod -Method Post -Uri $Url -Headers @{
  Authorization = "Bearer $Token"
  "Content-Type" = "application/json"
} -Body $body

$response | ConvertTo-Json -Depth 6

if ($response.error) {
  throw "Endpoint returned error: $($response.error)"
}

if ($response.used_fallback -and -not $AllowFallback.IsPresent) {
  throw "Endpoint used fallback audio. Set -AllowFallback only if you want fallback behavior."
}

if (-not $response.audio_base64_wav) {
  throw "No audio_base64_wav returned."
}

[IO.File]::WriteAllBytes($OutFile, [Convert]::FromBase64String($response.audio_base64_wav))
Write-Host "Saved audio file: $OutFile"
