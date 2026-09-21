# Visual Harness Biomechanics Basis

The visual harness is a **kinematic test generator**, not a clinical musculoskeletal simulator. It uses published exercise biomechanics to define nominal movement ranges and then enforces those movements with avatar-specific segment lengths and inverse kinematics.

Key design rules:

- Bench and overhead pressing are modeled as linked rigid segments constrained by the external load/bar path.
- Bench inclination uses 30° as the canonical incline condition because it is a well-studied incline condition.
- Bench grip defaults stay below 1.5 bi-acromial widths; grip width and elbow position materially alter shoulder/elbow mechanics.
- Squat and leg press are modeled as hip-knee-ankle chains. Bottom knee flexion is configurable rather than treated as a universal “correct” angle.
- Lateral raise uses a nominal 90° humeral elevation target with slight elbow flexion.
- Lat pulldown spans high-to-lower shoulder elevation and uses 1.2× shoulder width as a nominal hand spacing.
- Form errors are injected parametrically and measured back from the generated joint geometry.

Sources:

- https://pubmed.ncbi.nlm.nih.gov/20182386/
- https://pubmed.ncbi.nlm.nih.gov/33555823/
- https://pubmed.ncbi.nlm.nih.gov/35157403/
- https://pubmed.ncbi.nlm.nih.gov/38974522/
- https://pubmed.ncbi.nlm.nih.gov/25799093/
- https://pubmed.ncbi.nlm.nih.gov/33049982/
- https://pubmed.ncbi.nlm.nih.gov/34423289/
- https://pubmed.ncbi.nlm.nih.gov/30114973/
- https://pubmed.ncbi.nlm.nih.gov/24245055/
- https://pubmed.ncbi.nlm.nih.gov/24064179/
- https://threejs.org/docs/pages/CCDIKSolver.html
