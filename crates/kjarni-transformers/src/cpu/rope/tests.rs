use super::*;

#[cfg(test)]
mod rope_tests {
    use super::*;
    use ndarray::s;
    use ndarray::{Array3, Array4};
    /// Helper function to compare two 4D tensors for approximate equality.
    fn assert_tensors_approx_equal(a: &Array4<f32>, b: &Array4<f32>, tolerance: f32) {
        assert_eq!(a.shape(), b.shape(), "Tensor shapes do not match");
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (val_a - val_b).abs() < tolerance,
                "Tensor values differ at index {}: a={}, b={}. Difference: {}",
                i,
                val_a,
                val_b,
                (val_a - val_b).abs()
            );
        }
        println!("✓ Tensors are approximately equal.");
    }

    #[test]
    fn test_rope_pytorch_parity() {
        // --- 1. Arrange: Use the golden values from the Python script ---
        // Config matches the python script: batch=1, heads=2, seq_len=4, head_dim=8
        let head_dim = 8;
        let seq_len = 4;
        let position_offset = 10;
        let max_seq_len = position_offset + seq_len;
        let theta = 10000.0;

        // --- RoPE Input Q ---
        // Shape: [1, 2, 4, 8]
        let q_input_vec = vec![
            1.9269152879714966,
            1.4872840642929077,
            0.9007171988487244,
            -2.1055209636688232,
            0.6784184575080872,
            -1.2345448732376099,
            -0.04306747764348984,
            -1.6046669483184814,
            -0.7521352767944336,
            1.6487230062484741,
            -0.3924786448478699,
            -1.4036071300506592,
            -0.7278813123703003,
            -0.5594301819801331,
            -0.7688388824462891,
            0.7624453902244568,
            1.6423169374465942,
            -0.1595974713563919,
            -0.4973975419998169,
            0.439589262008667,
            -0.7581311464309692,
            1.078317642211914,
            0.8008005619049072,
            1.680620551109314,
            1.27912437915802,
            1.2964228391647339,
            0.610466480255127,
            1.334737777709961,
            -0.2316243201494217,
            0.041759490966796875,
            -0.2515752911567688,
            0.859858512878418,
            -1.3846737146377563,
            -0.8712361454963684,
            -0.223365917801857,
            1.7173614501953125,
            0.3188803195953369,
            -0.42451897263526917,
            0.3057209253311157,
            -0.7745925188064575,
            -1.5575724840164185,
            0.9956361055374146,
            -0.8797858357429504,
            -0.6011420488357544,
            -1.2741512060165405,
            2.1227850914001465,
            -1.234653115272522,
            -0.4879138767719269,
            -0.9138230085372925,
            -0.6581372618675232,
            0.07802387326955795,
            0.5258087515830994,
            -0.48799172043800354,
            1.1913690567016602,
            -0.8140076398849487,
            -0.7359927892684937,
            -1.4032478332519531,
            0.03600366786122322,
            -0.06347727030515671,
            0.6756148934364319,
            -0.0978068932890892,
            1.8445940017700195,
            -1.184537410736084,
            1.3835493326187134,
        ];
        let k_input_vec = vec![
            1.4451338052749634,
            0.8564125299453735,
            2.218075752258301,
            0.5231655240058899,
            0.34664666652679443,
            -0.19733144342899323,
            -1.0545889139175415,
            1.2779955863952637,
            -0.1721901297569275,
            0.5237884521484375,
            0.056621819734573364,
            0.4262961447238922,
            0.575005054473877,
            -0.6417241096496582,
            -2.2063984870910645,
            -0.7508030533790588,
            0.01086814422160387,
            -0.33874234557151794,
            -1.3406795263290405,
            -0.5853705406188965,
            0.5361881256103516,
            0.5246226191520691,
            1.1412016153335571,
            0.05164359509944916,
            0.7439519762992859,
            -0.4815843999385834,
            -1.0494661331176758,
            0.603898823261261,
            -1.7222950458526611,
            -0.827768862247467,
            1.334702968597412,
            0.48353928327560425,
            -2.5095443725585938,
            0.4880010485649109,
            0.7845868468284607,
            0.02864718623459339,
            0.640755295753479,
            0.5832474231719971,
            1.0669267177581787,
            -0.4501533806324005,
            -0.18526747822761536,
            0.7527588605880737,
            0.4047577977180481,
            0.17846599221229553,
            0.2649095058441162,
            1.2731683254241943,
            -0.0013108636485412717,
            -0.30360376834869385,
            -1.457029104232788,
            -0.10233523696660995,
            -0.5991530418395996,
            0.4770564138889313,
            0.7261772155761719,
            0.09115186333656311,
            -0.3890652060508728,
            0.5279164910316467,
            -0.012685478664934635,
            0.24083632230758667,
            0.13253536820411682,
            0.7642406225204468,
            1.095009684562683,
            0.3398909568786621,
            0.7199674844741821,
            0.41140761971473694,
        ];

        let q_cpu = Array4::from_shape_vec((1, 2, 4, 8), q_input_vec).unwrap();
        let k_cpu = Array4::from_shape_vec((1, 2, 4, 8), k_input_vec).unwrap();

        // --- RoPE Golden Output Q ---
        let expected_q_vec = vec![
            -1.2477457523345947,
            1.8424166440963745,
            0.900516927242279,
            -2.089369297027588,
            -1.6175241470336914,
            0.5844788551330566,
            0.047069355845451355,
            -1.6256415843963623,
            -0.7312029600143433,
            1.2464226484298706,
            -0.3057047128677368,
            -1.4119089841842651,
            0.7489065527915955,
            1.2155988216400146,
            -0.8072777390480042,
            0.7469598650932312,
            0.9790829420089722,
            -1.0628656148910522,
            -0.5896861553192139,
            0.41939064860343933,
            -1.52097487449646,
            0.24198561906814575,
            0.7354971766471863,
            1.6857744455337524,
            1.2580581903457642,
            0.30655381083488464,
            0.6379280090332031,
            1.3234471082687378,
            0.3272591531276703,
            1.2603495121002197,
            -0.17031517624855042,
            0.8771369457244873,
            1.3353179693222046,
            -0.11351054906845093,
            -0.25277116894721985,
            1.7250213623046875,
            0.48572838306427,
            -0.9624885320663452,
            0.2818942070007324,
            -0.7573804259300232,
            -1.2810322046279907,
            -1.4402251243591309,
            -0.7389304041862488,
            -0.5957387685775757,
            1.5519182682037354,
            1.8502053022384644,
            -1.3237723112106323,
            -0.49449679255485535,
            -1.0329763889312744,
            -1.3488836288452148,
            0.17490942776203156,
            0.5346025824546814,
            0.07853895425796509,
            -0.18170788884162903,
            -0.7988134026527405,
            -0.7296302318572998,
            -1.232277512550354,
            -1.7677427530288696,
            0.09061485528945923,
            0.6575721502304077,
            -0.6783530116081238,
            0.5281182527542114,
            -1.1827709674835205,
            1.3922151327133179,
        ];
        let expected_k_vec = vec![
            -1.0239874124526978,
            0.6287703514099121,
            2.3122777938842773,
            0.5103595852851868,
            -1.0770446062088013,
            0.6140276193618774,
            -0.8278822898864746,
            1.28316330909729,
            0.5742374062538147,
            0.8094976544380188,
            0.29849427938461304,
            0.43452903628349304,
            0.1747332513332367,
            0.17572057247161865,
            -2.186847448348999,
            -0.7460684776306152,
            0.29687514901161194,
            -0.6117147207260132,
            -1.4676539897918701,
            -0.58594810962677,
            0.44663292169570923,
            -0.1256200522184372,
            0.9724990725517273,
            0.044615600258111954,
            1.3987483978271484,
            0.6687802672386169,
            -1.2136337757110596,
            0.5975619554519653,
            -1.2503070831298828,
            -0.6854617595672607,
            1.1873939037322998,
            0.49134889245033264,
            2.4542715549468994,
            -0.227117657661438,
            0.6741522550582886,
            0.033147212117910385,
            0.8276057243347168,
            0.7257686853408813,
            1.1399245262145996,
            -0.4498444199562073,
            0.2640869617462158,
            -0.7932085394859314,
            0.4024553894996643,
            0.18179476261138916,
            0.18643806874752045,
            1.248368501663208,
            0.04313068091869354,
            -0.30162233114242554,
            -0.8398727178573608,
            -0.1220390647649765,
            -0.5482684969902039,
            0.47068721055984497,
            1.394589900970459,
            -0.062350861728191376,
            -0.4579932391643524,
            0.5336030125617981,
            -0.471598356962204,
            -0.26308131217956543,
            0.03808465600013733,
            0.7588278651237488,
            0.988332986831665,
            0.32298022508621216,
            0.7310734391212463,
            0.42130768299102783,
        ];

        let expected_q_cpu = Array4::from_shape_vec((1, 2, 4, 8), expected_q_vec).unwrap();
        let expected_k_cpu = Array4::from_shape_vec((1, 2, 4, 8), expected_k_vec).unwrap();

        // --- 2. Act ---
        let rope = RoPE::new(head_dim, max_seq_len, theta);
        let (actual_q_cpu, actual_k_cpu) = rope.apply_4d(&q_cpu, &k_cpu, position_offset);

        // --- 3. Assert ---
        let tolerance = 1e-5; // Slightly looser for RoPE due to sin/cos
        assert_tensors_approx_equal(&expected_q_cpu, &actual_q_cpu, tolerance);
        assert_tensors_approx_equal(&expected_k_cpu, &actual_k_cpu, tolerance);

        println!("✓ CPU RoPE implementation passed PyTorch parity test.");
    }
    #[test]
    fn test_rope_actually_rotates() {
        let head_dim = 64;
        let max_seq_len = 128;
        let rope = RoPE::new(head_dim, max_seq_len, 10000.0);

        // Create simple input
        let q = Array4::ones((1, 4, 8, head_dim)); // [batch, heads, seq, dim]
        let k = Array4::ones((1, 4, 8, head_dim));

        let (q_rotated, k_rotated) = rope.apply_4d(&q, &k, 0);

        // Check that rotation actually changed values
        let q_diff = (&q - &q_rotated).mapv(|x| x.abs()).sum();
        let k_diff = (&k - &k_rotated).mapv(|x| x.abs()).sum();

        assert!(
            q_diff > 1e-3,
            "Q should be modified by RoPE, diff={}",
            q_diff
        );
        assert!(
            k_diff > 1e-3,
            "K should be modified by RoPE, diff={}",
            k_diff
        );

        println!("Q diff: {}", q_diff);
        println!("K diff: {}", k_diff);
    }

    #[test]
    fn test_rope_different_positions() {
        let head_dim = 64;
        let max_seq_len = 128;
        let rope = RoPE::new(head_dim, max_seq_len, 10000.0);

        let q = Array4::from_shape_fn((1, 4, 1, head_dim), |(_, _, _, i)| i as f32);
        let k = Array4::from_shape_fn((1, 4, 1, head_dim), |(_, _, _, i)| i as f32);

        let (q_pos0, k_pos0) = rope.apply_4d(&q, &k, 0);
        let (q_pos5, k_pos5) = rope.apply_4d(&q, &k, 5);

        let q_diff = (&q_pos0 - &q_pos5).mapv(|x| x.abs()).sum();
        let k_diff = (&k_pos0 - &k_pos5).mapv(|x| x.abs()).sum();

        assert!(
            q_diff > 1e-3,
            "RoPE should give different results for different positions, Q diff={}",
            q_diff
        );
        assert!(
            k_diff > 1e-3,
            "RoPE should give different results for different positions, K diff={}",
            k_diff
        );
    }

    #[test]
    fn test_rope_frequencies() {
        let head_dim = 64;
        let rope = RoPE::new(head_dim, 128, 10000.0);
        assert_eq!(rope.cos_cache.shape(), &[128, head_dim]);
        assert_eq!(rope.sin_cache.shape(), &[128, head_dim]);

        // First cos value should be close to cos(0) = 1.0
        let first_cos = rope.cos_cache[[0, 0]];
        assert!((first_cos - 1.0).abs() < 1e-6, "First cos should be 1.0");

        // First sin value should be close to sin(0) = 0.0
        let first_sin = rope.sin_cache[[0, 0]];
        assert!((first_sin - 0.0).abs() < 1e-6, "First sin should be 0.0");
    }

    #[test]
    fn test_rope_precompute() {
        let head_dim = 4;
        let max_seq_len = 8;
        let theta = 10000.0;

        let rope = RoPE::new(head_dim, max_seq_len, theta);

        // ✅ Cache shape now correctly matches full head_dim
        assert_eq!(rope.cos_cache.shape(), &[max_seq_len, head_dim]);
        assert_eq!(rope.sin_cache.shape(), &[max_seq_len, head_dim]);

        // Check that values are finite
        for i in 0..max_seq_len {
            for j in 0..head_dim {
                assert!(rope.cos_cache[[i, j]].is_finite());
                assert!(rope.sin_cache[[i, j]].is_finite());
            }
        }
    }
    #[test]
    fn test_rope_simple_debug() {
        let head_dim = 4;
        let rope = RoPE::new(head_dim, 10, 10000.0);

        println!("\n=== RoPE Debug Info ===");
        println!("head_dim: {}", rope.head_dim());

        // ✅ Use position 1, not 0
        let q = Array4::ones((1, 1, 1, head_dim));
        let k = Array4::ones((1, 1, 1, head_dim));

        println!("Input Q: {:?}", q.slice(s![0, 0, 0, ..]));

        // Test at position 1 (not 0, which is identity)
        let (q_rot, _) = rope.apply_4d(&q, &k, 1);

        println!("Rotated Q at pos=1: {:?}", q_rot.slice(s![0, 0, 0, ..]));

        let sum_diff = (&q - &q_rot).mapv(|x| x.abs()).sum();
        println!("Total difference: {}", sum_diff);

        assert!(
            sum_diff > 1e-3,
            "RoPE should change values at pos=1, but diff={}",
            sum_diff
        );
    }
    #[test]
    fn test_rope_rotation_formula() {
        let head_dim = 4;
        let rope = RoPE::new(head_dim, 10, 10000.0);

        let mut q = Array4::zeros((1, 1, 1, head_dim));
        q[[0, 0, 0, 0]] = 1.0;
        q[[0, 0, 0, 1]] = 2.0;
        q[[0, 0, 0, 2]] = 3.0;
        q[[0, 0, 0, 3]] = 4.0;

        let k = q.clone();
        let (q_rot, _) = rope.apply_4d(&q, &k, 1);

        let changed = (q[[0, 0, 0, 0]] - q_rot[[0, 0, 0, 0]]).abs() > 1e-6;
        assert!(changed, "RoPE should modify values at position 1");
    }
    #[test]
    fn test_rope_basic_rotation() {
        use ndarray::Array4;

        let head_dim = 64;
        let rope = RoPE::new(head_dim, 128, 10000.0);
        let half_dim = head_dim / 2;

        // Simple test vector:
        // q[0] = 1.0
        // q[32] (its pair in the other half) = 2.0
        // All other elements are 0.0
        let mut q = Array4::zeros((1, 1, 1, head_dim));
        q[[0, 0, 0, 0]] = 1.0;
        q[[0, 0, 0, half_dim]] = 2.0;

        let k = q.clone();

        println!("\n=== Testing RoPE Rotation (Corrected) ===");
        println!(
            "Input Q[0]: {}, Q[{}]: {}",
            q[[0, 0, 0, 0]],
            half_dim,
            q[[0, 0, 0, half_dim]]
        );

        // Rotate at position 5
        let (q_rot, _) = rope.apply_4d(&q, &k, 5);

        println!(
            "Rotated Q[0]: {}, Rotated Q[{}]: {}",
            q_rot[[0, 0, 0, 0]],
            half_dim,
            q_rot[[0, 0, 0, half_dim]]
        );

        // Get expected values from the cache for the first dimension pair (i=0)
        let cos_5_0 = rope.cos_cache[[5, 0]];
        let sin_5_0 = rope.sin_cache[[5, 0]];

        println!("cos[5,0]: {}, sin[5,0]: {}", cos_5_0, sin_5_0);
        // Formula for q_rot[i] = q[i] * cos[i] + (-q[i + half_dim]) * sin[i]
        // Formula for q_rot[i + half_dim] = q[i + half_dim] * cos[i] + q[i] * sin[i]

        let q0 = q[[0, 0, 0, 0]]; // is 1.0
        let q32 = q[[0, 0, 0, half_dim]]; // is 2.0

        let expected_0 = q0 * cos_5_0 - q32 * sin_5_0; // 1.0 * cos - 2.0 * sin
        let expected_32 = q32 * cos_5_0 + q0 * sin_5_0; // 2.0 * cos + 1.0 * sin

        println!(
            "Expected Q[0]: {}, Expected Q[{}]: {}",
            expected_0, half_dim, expected_32
        );

        let diff_0 = (q_rot[[0, 0, 0, 0]] - expected_0).abs();
        let diff_32 = (q_rot[[0, 0, 0, half_dim]] - expected_32).abs();

        assert!(
            diff_0 < 1e-5,
            "Q[0] should match expected: {} vs {}. Diff: {}",
            q_rot[[0, 0, 0, 0]],
            expected_0,
            diff_0
        );
        assert!(
            diff_32 < 1e-5,
            "Q[{}] should match expected: {} vs {}. Diff: {}",
            half_dim,
            q_rot[[0, 0, 0, half_dim]],
            expected_32,
            diff_32
        );

        println!("✓ RoPE rotation correct!");
    }

    #[test]
    fn test_rope_rotation_basic() {
        let head_dim = 4;
        let max_seq_len = 8;
        let theta = 10000.0;

        let rope = RoPE::new(head_dim, max_seq_len, theta);

        // Simple query and key: [1, 1, 1, 4]
        let q = Array4::from_shape_vec((1, 1, 1, 4), vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        let k = Array4::from_shape_vec((1, 1, 1, 4), vec![0.0, 1.0, 0.0, 1.0]).unwrap();

        let (rotated_q, rotated_k) = rope.apply_4d(&q, &k, 0);

        // Check output shapes
        assert_eq!(rotated_q.shape(), &[1, 1, 1, 4]);
        assert_eq!(rotated_k.shape(), &[1, 1, 1, 4]);

        // Check values are finite
        assert!(rotated_q[[0, 0, 0, 0]].is_finite());
        assert!(rotated_k[[0, 0, 0, 0]].is_finite());
    }

    #[test]
    fn test_rope_rotation_identity() {
        // At position 0, rotation should be minimal (close to identity)
        let head_dim = 4;
        let max_seq_len = 8;
        let theta = 10000.0;

        let rope = RoPE::new(head_dim, max_seq_len, theta);

        let q = Array4::from_shape_vec((1, 1, 1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let k = q.clone();

        let (rotated_q, _) = rope.apply_4d(&q, &k, 0);

        // At position 0, first dimension pair should be very close to original
        // cos(0) = 1, sin(0) = 0, so rotation is identity
        assert!((rotated_q[[0, 0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((rotated_q[[0, 0, 0, 1]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_position_offset() {
        let head_dim = 4;
        let max_seq_len = 16;
        let theta = 10000.0;

        let rope = RoPE::new(head_dim, max_seq_len, theta);

        let q = Array4::from_shape_vec(
            (1, 1, 2, 4),
            vec![
                1.0, 0.0, 1.0, 0.0, // position 0 (or offset)
                1.0, 0.0, 1.0, 0.0, // position 1 (or offset+1)
            ],
        )
        .unwrap();
        let k = q.clone();

        // Apply with offset 0
        let (rotated_q_0, _) = rope.apply_4d(&q, &k, 0);

        // Apply with offset 5
        let (rotated_q_5, _) = rope.apply_4d(&q, &k, 5);

        // Values should be different because positions are different
        assert!((rotated_q_0[[0, 0, 0, 0]] - rotated_q_5[[0, 0, 0, 0]]).abs() > 1e-3);
    }

    #[test]
    fn test_rope_3d_interface() {
        let head_dim = 4;
        let num_heads = 2;
        let hidden_size = head_dim * num_heads;
        let max_seq_len = 8;
        let theta = 10000.0;

        let rope = RoPE::new(head_dim, max_seq_len, theta);

        // [batch=1, seq_len=2, hidden_size=8]
        let q = Array3::from_shape_vec(
            (1, 2, hidden_size),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // pos 0
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // pos 1
            ],
        )
        .unwrap();
        let k = q.clone();

        // FIX: num_kv_heads must be > 0. For non-GQA, it's equal to num_q_heads.
        let num_kv_heads = num_heads;
        let result = rope.apply_3d(&q, &k, num_heads, num_kv_heads, 0);

        assert!(result.is_ok(), "RoPE apply_3d should succeed");
        let (rotated_q, rotated_k) = result.unwrap();

        // Check output shape preserved
        assert_eq!(rotated_q.shape(), &[1, 2, hidden_size]);
        assert_eq!(rotated_k.shape(), &[1, 2, hidden_size]);
    }
    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation, so it should preserve L2 norm
        let head_dim = 4;
        let max_seq_len = 8;
        let theta = 10000.0;

        let rope = RoPE::new(head_dim, max_seq_len, theta);

        let q = Array4::from_shape_vec((1, 1, 1, 4), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let k = q.clone();

        // Original norm: sqrt(9 + 16) = 5.0
        let original_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();

        let (rotated_q, _) = rope.apply_4d(&q, &k, 0);

        let rotated_norm: f32 = rotated_q.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Norm should be preserved (within floating point tolerance)
        assert!((original_norm - rotated_norm).abs() < 1e-4);
    }

    #[test]
    fn test_rope_pytorch_parity_2() {
        // Test against known PyTorch output
        // PyTorch code:
        // ```python
        // import torch
        // def precompute_freqs_cis(dim, end, theta=10000.0):
        //     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        //     t = torch.arange(end)
        //     freqs = torch.outer(t, freqs).float()
        //     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        //     return freqs_cis
        //
        // def apply_rotary_emb(xq, xk, freqs_cis):
        //     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        //     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        //     freqs_cis = freqs_cis[:xq_.shape[1]]
        //     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        //     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        //     return xq_out, xk_out
        //
        // freqs = precompute_freqs_cis(4, 8)
        // q = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        // k = torch.tensor([[[[0.0, 1.0, 0.0, 1.0]]]])
        // rq, rk = apply_rotary_emb(q, k, freqs)
        // print(rq, rk)
        // ```

        let head_dim = 4;
        let max_seq_len = 8;
        let theta = 10000.0;

        let rope = RoPE::new(head_dim, max_seq_len, theta);

        let q = Array4::from_shape_vec((1, 1, 1, 4), vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        let k = Array4::from_shape_vec((1, 1, 1, 4), vec![0.0, 1.0, 0.0, 1.0]).unwrap();

        let (rotated_q, _) = rope.apply_4d(&q, &k, 0);

        // At position 0:
        // freq[0] = 1.0 / (10000^(0/4)) = 1.0, angle = 0 * 1.0 = 0
        // freq[1] = 1.0 / (10000^(2/4)) = 0.01, angle = 0 * 0.01 = 0
        // So cos ≈ [1.0, 1.0, 1.0, 1.0], sin ≈ [0.0, 0.0, 0.0, 0.0]

        // For q = [1, 0, 1, 0]:
        // Pair 0: [1*1 - 0*0, 1*0 + 0*1] = [1, 0]
        // Pair 1: [1*1 - 0*0, 1*0 + 0*1] = [1, 0]
        assert!((rotated_q[[0, 0, 0, 0]] - 1.0).abs() < 1e-3);
        assert!((rotated_q[[0, 0, 0, 1]] - 0.0).abs() < 1e-3);
        assert!((rotated_q[[0, 0, 0, 2]] - 1.0).abs() < 1e-3);
        assert!((rotated_q[[0, 0, 0, 3]] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_rope_multiple_positions() {
        let head_dim = 8;
        let max_seq_len = 16;
        let theta = 10000.0;

        let rope = RoPE::new(head_dim, max_seq_len, theta);

        // Test with a sequence of length 3
        let q = Array4::from_shape_vec(
            (1, 1, 3, 8),
            vec![
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // pos 0
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // pos 1
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // pos 2
            ],
        )
        .unwrap();
        let k = q.clone();

        let (rotated_q, _) = rope.apply_4d(&q, &k, 0);

        // Each position should have different rotations
        let pos0_val = rotated_q[[0, 0, 0, 0]];
        let pos1_val = rotated_q[[0, 0, 1, 0]];
        let pos2_val = rotated_q[[0, 0, 2, 0]];

        assert!((pos0_val - pos1_val).abs() > 1e-5);
        assert!((pos1_val - pos2_val).abs() > 1e-5);
    }
    #[test]
    fn test_rope_is_working() {
        use ndarray::{Array4, s};

        let head_dim = 16;
        let rope = RoPE::new(head_dim, 128, 10000.0);

        // Simple input
        let q = Array4::ones((1, 1, 1, head_dim));
        let k = Array4::ones((1, 1, 1, head_dim));

        println!("\n=== Testing RoPE ===");
        println!("Input Q: {:?}", q.slice(s![0, 0, 0, ..]));

        // Apply RoPE at position 1 (not 0, which is identity)
        let (q_rot, _) = rope.apply_4d(&q, &k, 1);

        println!("Rotated Q: {:?}", q_rot.slice(s![0, 0, 0, ..]));

        let diff = (&q - &q_rot).mapv(|x| x.abs()).sum();
        println!("Difference: {}", diff);

        assert!(
            diff > 0.01,
            "RoPE should change values at position 1, diff={}",
            diff
        );
    }
}
